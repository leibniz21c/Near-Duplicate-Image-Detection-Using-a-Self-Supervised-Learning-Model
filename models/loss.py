import torch
import torch.nn.functional as F

from torch import nn


class NT_Xent(nn.Module):
    """Algorithm from: https://github.com/google-research/simclr/blob/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3/tf2/objective.py # noqa: E501
    (accessed 13.11.2021) translated to pytorch
    """
    def __init__(self, temperature=1.0):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.device = None

    def forward(self, z_i, z_j):
        mask = torch.diag(torch.ones(z_i.shape[0])) * 1e9
        labels = F.one_hot(
            torch.arange(z_i.shape[0]),
            num_classes=z_i.shape[0] * 2,
        )
        mask = mask.to(self.device)
        labels = labels.to(self.device)

        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        sim_ii = torch.mm(z_i, z_i.t()) / self.temperature
        sim_ii = sim_ii - mask
        sim_jj = torch.mm(z_j, z_j.t()) / self.temperature
        sim_jj = sim_jj - mask

        sim_ij = torch.mm(z_i, z_j.t()) / self.temperature
        sim_ji = torch.mm(z_j, z_i.t()) / self.temperature

        sim_i = torch.cat([sim_ij, sim_ii], 1)
        sim_j = torch.cat([sim_ji, sim_jj], 1)

        loss_i = torch.sum(- labels * F.log_softmax(sim_i, -1))
        loss_j = torch.sum(- labels * F.log_softmax(sim_j, -1))

        loss = loss_i + loss_j

        loss /= 2 * z_i.shape[0]

        return loss