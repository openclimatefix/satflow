import numpy as np
from collections import abc
import torch
from typing import Mapping


def restructure(modality_sizes, inputs: np.ndarray) -> Mapping[str, np.ndarray]:
    """Partitions a [B, N, C] tensor into tensors for each modality.
    Args:
      modality_sizes: dict specifying the size of the modality
      inputs: input tensor
    Returns:
      dict mapping name of modality to its associated tensor.
    """
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


class AbstractPerceiverDecoder(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, query, z, *, is_training, query_mask=None):
        raise NotImplementedError


class ProjectionDecoder(AbstractPerceiverDecoder):
    """Baseline projection decoder (no cross-attention)."""

    def __init__(self, num_classes, final_avg_before_project=False, name="projection_decoder"):
        super().__init__()
        self._final_avg_before_project = final_avg_before_project
        self._num_classes = num_classes
        self.final_layer = torch.nn.Linear(num_classes)

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return None

    def output_shape(self, inputs):
        return inputs.shape[0], self._num_classes, None

    def __call__(self, query, z, *, is_training, query_mask=None):
        # b x n_z x c -> b x c
        z = np.mean(z, axis=1, dtype=z.dtype)
        # b x c -> b x n_logits
        logits = self.final_layer(z)
        return logits


class BasicDecoder(AbstractPerceiverDecoder):
    """Cross-attention-based decoder."""

    def __init__(
        self,
        output_num_channels,
        position_encoding_type="trainable",
        # Ignored if position_encoding_type == 'none':
        output_index_dims=None,
        subsampled_index_dims=None,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        use_query_residual=False,
        output_w_init=None,
        concat_preprocessed_input=False,
        num_heads=1,
        name="basic_decoder",
        final_project=True,
        **position_encoding_kwargs,
    ):
        super().__init__(name=name)
        self._position_encoding_type = position_encoding_type

        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_pos_enc = None
        if self._position_encoding_type != "none":
            self.output_pos_enc = position_encoding.build_position_encoding(
                position_encoding_type, index_dims=output_index_dims, **position_encoding_kwargs
            )

        self._output_index_dim = output_index_dims
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self._subsampled_index_dims = subsampled_index_dims
        self._output_num_channels = output_num_channels
        self._output_w_init = output_w_init
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._final_project = final_project
        self._num_heads = num_heads

        self._concat_preprocessed_input = concat_preprocessed_input

    def output_shape(self, inputs):
        return ((inputs[0], self._subsampled_index_dims, self._output_num_channels), None)

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        assert self._position_encoding_type != "none"  # Queries come from elsewhere
        if subsampled_points is not None:
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            pos = np.stack(np.unravel_index(subsampled_points, self._output_index_dim), axis=1)
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / np.array(self._output_index_dim)[None, :]
            pos = np.broadcast_to(pos[None], [inputs.shape[0], pos.shape[0], pos.shape[1]])
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0], pos=pos)
            pos_emb = np.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0])
        if self._concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError(
                    "Value is required for inputs_without_pos if"
                    " concat_preprocessed_input is True"
                )
            pos_emb = np.concatenate([inputs_without_pos, pos_emb], axis=-1)

        return pos_emb

    def __call__(self, query, z, *, is_training, query_mask=None):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # Construct cross attention and linear layer lazily, in case we don't need
        # them.
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask, kv_mask=np.ones(z.shape[:2], dtype=np.int32)
            )
        decoding_cross_attn = CrossAttention(
            dropout_prob=0.0,
            num_heads=self._num_heads,
            widening_factor=1,
            shape_for_attn="kv",
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=self._use_query_residual,
        )
        final_layer = torch.nn.Linear(
            self._output_num_channels, w_init=self._output_w_init, name="output"
        )
        output = decoding_cross_attn(
            query, z, is_training=is_training, attention_mask=attention_mask
        )
        if self._final_project:
            output = final_layer(output)
        return output


class ClassificationDecoder(AbstractPerceiverDecoder):
    """Cross-attention based classification decoder.

    Light-weight wrapper of `BasicDecoder` for logit output.
    """

    def __init__(self, num_classes, name="classification_decoder", **decoder_kwargs):
        super().__init__(name=name)

        self._num_classes = num_classes
        self.decoder = BasicDecoder(
            output_index_dims=(1,),  # Predict a single logit array.
            output_num_channels=num_classes,
            **decoder_kwargs,
        )

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def output_shape(self, inputs):
        return (inputs.shape[0], self._num_classes), None

    def __call__(self, query, z, *, is_training, query_mask=None):
        # B x 1 x num_classes -> B x num_classes
        logits = self.decoder(query, z, is_training=is_training)
        return logits[:, 0, :]


class MultimodalDecoder(AbstractPerceiverDecoder):
    """Multimodal decoding by composing uni-modal decoders.

    The modalities argument of the constructor is a dictionary mapping modality
    name to the decoder of that modality. That decoder will be used to construct
    queries for that modality. However, there is a shared cross attention across
    all modalities, using the concatenated per-modality query vectors.
    """

    def __init__(
        self,
        modalities,
        num_outputs,
        output_num_channels,
        min_padding_size=2,
        subsampled_index_dims=None,
        name="multimodal_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)
        self._modalities = modalities
        self._subsampled_index_dims = subsampled_index_dims
        self._min_padding_size = min_padding_size
        self._output_num_channels = output_num_channels
        self._num_outputs = num_outputs
        self._decoder = BasicDecoder(
            output_index_dims=(num_outputs,),
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            **decoder_kwargs,
        )

    def decoder_query(
        self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None
    ):
        # Partition the flat inputs among the different modalities
        inputs = io_processors.restructure(modality_sizes, inputs)
        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()
        decoder_queries = dict()
        for modality, decoder in self._modalities.items():
            # Get input_without_pos for this modality if it exists.
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            decoder_queries[modality] = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=input_without_pos,
                subsampled_points=subsampled_points.get(modality, None),
            )

        # Pad all queries with trainable position encodings to make them
        # have the same channels
        num_channels = (
            max(query.shape[2] for query in decoder_queries.values()) + self._min_padding_size
        )

        def embed(modality, x):
            x = np.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = position_encoding.TrainablePositionEncoding(
                1,
                num_channels=num_channels - x.shape[2],
                init_scale=0.02,
                name=f"{modality}_padding",
            )(x.shape[0])
            pos = np.broadcast_to(pos, [x.shape[0], x.shape[1], num_channels - x.shape[2]])
            return np.concatenate([x, pos], axis=2)

        # Apply a predictable ordering to the modalities
        return np.concatenate(
            [
                embed(modality, decoder_queries[modality])
                for modality in sorted(self._modalities.keys())
            ],
            axis=1,
        )

    def output_shape(self, inputs):
        if self._subsampled_index_dims is not None:
            subsampled_index_dims = sum(self._subsampled_index_dims.values())
        else:
            subsampled_index_dims = self._num_outputs
        return (
            (inputs.shape[0], subsampled_index_dims, self._output_num_channels),
            self._subsampled_index_dims,
        )

    def __call__(self, query, z, *, is_training, query_mask=None):
        # B x 1 x num_classes -> B x num_classes
        return self._decoder(query, z, is_training=is_training)


class BasicVideoAutoencodingDecoder(AbstractPerceiverDecoder):
    """Cross-attention based video-autoencoding decoder.

    Light-weight wrapper of `BasicDecoder` with video reshaping logic.
    """

    def __init__(
        self,
        output_shape,
        position_encoding_type,
        name="basic_video_autoencoding_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)
        if len(output_shape) != 4:  # B, T, H, W
            raise ValueError(f"Expected rank 4 output_shape, got {output_shape}.")
        # Build the decoder components:
        self._output_shape = output_shape
        self._output_num_channels = decoder_kwargs["output_num_channels"]

        self.decoder = BasicDecoder(
            output_index_dims=self._output_shape[1:4],  # T*H*W
            position_encoding_type=position_encoding_type,
            **decoder_kwargs,
        )

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return self.decoder.decoder_query(
            inputs,
            modality_sizes=modality_sizes,
            inputs_without_pos=inputs_without_pos,
            subsampled_points=subsampled_points,
        )

    def output_shape(self, inputs):
        return [inputs.shape[0]] + self._output_shape[1:] + [self._output_num_channels], None

    def __call__(self, query, z, *, is_training, query_mask=None):
        output = self.decoder(query, z, is_training=is_training)

        output = np.reshape(output, self._output_shape + [output.shape[-1]])
        return output


class FlowDecoder(AbstractPerceiverDecoder):
    """Cross-attention based flow decoder."""

    def __init__(
        self,
        output_image_shape,
        output_num_channels=2,
        rescale_factor=100.0,
        name="flow_decoder",
        **decoder_kwargs,
    ):
        super().__init__(name=name)

        self._output_image_shape = output_image_shape
        self._output_num_channels = output_num_channels
        self._rescale_factor = rescale_factor
        self.decoder = BasicDecoder(output_num_channels=output_num_channels, **decoder_kwargs)

    def output_shape(self, inputs):
        # The channel dimensions of output here don't necessarily correspond to
        # (u, v) of flow: they may contain dims needed for the post-processor.
        return (
            (inputs.shape[0],) + tuple(self._output_image_shape) + (self._output_num_channels,),
            None,
        )

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        # assumes merged in time
        return inputs

    def __call__(self, query, z, *, is_training, query_mask=None):
        # Output flow and rescale.
        preds = self.decoder(query, z, is_training=is_training)
        preds /= self._rescale_factor

        return preds.reshape([preds.shape[0]] + list(self._output_image_shape) + [preds.shape[-1]])
