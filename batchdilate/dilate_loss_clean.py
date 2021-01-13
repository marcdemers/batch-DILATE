import torch
from torch import Tensor

from . import dtw
from . import path_dtw2


class DTWShpTime(torch.nn.Module):
    def __init__(self, alpha, gamma):
        """
        Batch-DILATE loss function, a batchwise extension of https://github.com/vincent-leguen/DILATE

        :param alpha: Weight of shape component of the loss versus the temporal component.
        :type alpha: float
        :param gamma: Weight of softmax component of DTW.
        :type gamma: float
        """
        super(DTWShpTime, self).__init__()
        assert 0 <= alpha <= 1
        assert 0 <= gamma <= 1
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Pass through the loss function with input tensor (the prediction) and the target tensor.

        :param input: prediction, shape should be (batch, channels, num_timesteps_outputs).
        :type input: torch.Tensor
        :param target: target with same shape as prediction.
        :type target: torch.Tensor
        :return: total_loss, shape_loss, temporal_loss, with first dimensions being the batch
        :rtype: tuple
        """
        assert input.device == target.device
        batch_size, N_channel, N_output = input.shape

        D = dtw.pairwise_distances_with_channels_and_batches(
            target[:, :, :].reshape(batch_size * N_channel, N_output, 1).double(),
            input[:, :, :].reshape(batch_size * N_channel, N_output, 1).double()
        )

        D = D.reshape(batch_size, N_channel, N_output, N_output)

        softdtw_batch = dtw.SoftDTWBatch.apply
        loss_shape = softdtw_batch(D, self.gamma)

        path_dtw = path_dtw2.PathDTWBatch2.apply
        path = path_dtw(D, self.gamma)

        Omega = dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(target.device)

        Omega = Omega.repeat(N_channel, 1, 1)
        loss_temporal = torch.sum(path * Omega, dim=(1, 2)) / (N_output * N_output)

        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        loss = loss.mean()

        return loss, loss_shape, loss_temporal
