## Baseline

To see if our ML models are actually improving the predictions of the cloud masks, we
have benchmarked it against OpenCV's dense optical flow predictions, as well as a naive
baseline of just predicting the current image for all future timesteps.

As we come up with and implement more metrics to compare these models, they will be added
here. Currently, the only metric tested is the mean squared error between the predicted frame
and the ground truth frame. To get a sense if there is a temporal dependence, the mean loss is
done not just for the overall predictions, but for each of the future timesteps, going up to 4 hours (48 timesteps)
in the future.

On average, the optical flow approach has an MSE of 0.1541. The naive baseline has a MSE of 0.1566,
so optical flow beats out the naive baseline by about 1.6%.

## Caveats

We tried obtaining the optical flow of consecutive, or even very temporally separated the cloud masks,
but the optical flow usually ended up not actually changing anything. Instead, we used the
MSG HRV satellite channel to compute the optical flow. This was chosen as that is the highest
resolution satellite channel available, and it resulted in optical flow actually computing some movement.
This flow field was then applied to the cloud masks directly to obtain the flow results.

Avg Total Loss: 0.15720261434381796 Avg Baseline Loss: 0.1598897848692671
Overall Loss: 0.15720261434381738 Baseline: 0.1598897848692671
