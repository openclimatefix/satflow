"""
class Perceiver(hk.Module):
    """ """

    def __init__(
        self,
        encoder,
        decoder,
        input_preprocessor=None,
        output_postprocessor=None,
        name="perceiver",
    ):
        super().__init__(name=name)

        # Feature and task parameters:
        self._input_preprocessor = input_preprocessor
        self._output_postprocessor = output_postprocessor
        self._decoder = decoder
        self._encoder = encoder

    def __call__(
        self,
        inputs,
        *,
        is_training,
        subsampled_output_points=None,
        pos=None,
        input_mask=None,
        query_mask=None,
    ):
        if self._input_preprocessor:
            network_input_is_1d = self._encoder._input_is_1d
            inputs, modality_sizes, inputs_without_pos = self._input_preprocessor(
                inputs, pos=pos, is_training=is_training, network_input_is_1d=network_input_is_1d
            )
        else:
            modality_sizes = None
            inputs_without_pos = None

        # Get the queries for encoder and decoder cross-attends.
        encoder_query = self._encoder.latents(inputs)
        decoder_query = self._decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points
        )

        # Run the network forward:
        z = self._encoder(inputs, encoder_query, is_training=is_training, input_mask=input_mask)
        _, output_modality_sizes = self._decoder.output_shape(inputs)
        output_modality_sizes = output_modality_sizes or modality_sizes

        outputs = self._decoder(decoder_query, z, is_training=is_training, query_mask=query_mask)

        if self._output_postprocessor:
            outputs = self._output_postprocessor(
                outputs, is_training=is_training, modality_sizes=output_modality_sizes
            )

        return outputs


class PerceiverEncoder(hk.Module):
    #The Perceiver Encoder: a scalable, fully attentional encoder.

    def __init__(
        self,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn="kv",
        use_query_residual=True,
        name="perceiver_encoder",
    ):
        super().__init__(name=name)

        # Check that we can use multihead-attention with these shapes.
        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({num_z_channels}) must be divisible by"
                f" num_self_attend_heads ({num_self_attend_heads})."
            )
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(
                f"num_z_channels ({num_z_channels}) must be divisible by"
                f" num_cross_attend_heads ({num_cross_attend_heads})."
            )

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.z_pos_enc = position_encoding.TrainablePositionEncoding(
            index_dim=z_index_dim, num_channels=num_z_channels, init_scale=z_pos_enc_init_scale
        )

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            dropout_prob=dropout_prob,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            shape_for_attn=cross_attention_shape_for_attn,
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=use_query_residual,
        )

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = []
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                widening_factor=self_attend_widening_factor,
            )
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def __call__(self, inputs, z, *, is_training, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=jnp.ones(z.shape[:2], dtype=jnp.int32), kv_mask=input_mask
            )
        z = self.cross_attend(z, inputs, is_training=is_training, attention_mask=attention_mask)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z, is_training=is_training)
        return z
"""

"""
Taken from https://github.com/fac2003/perceiver-multi-modality-pytorch/tree/main/perceiver_pytorch

"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InputModality:
    name: str
    input_channels: int
    input_axis: int
    num_freq_bands: int
    max_freq: float
    freq_base: int = 2

    @property
    def input_dim(self) -> int:
        # Calculate the dimension of this modality.
        input_dim = self.input_axis * ((self.num_freq_bands * 2) + 1) + self.input_channels
        return input_dim


def modality_encoding(batch_size: int, axes, modality_index: int, num_modalities: int) -> Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.
    :param modality_index:
    :param num_modalities:
    :return:
    """
    one_hot = torch.eye(num_modalities, num_modalities)[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    to_expand.append(num_modalities)

    one_hot = one_hot.expand(to_expand)
    return one_hot
