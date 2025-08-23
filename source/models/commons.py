import torch
from torch import nn
import torch.nn.functional as F


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    weight_init="xavier",
):
    m = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
            weight_init="kaiming",
        )

    def forward(self, x):
        x = self.m(x)
        return F.relu(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, obs_size: int, obs_channels: int):
        super().__init__()
        width = height = obs_size
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, obs_channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x):
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x


class BroadCastDecoder(nn.Module):
    def __init__(self, obs_size, obs_channels, hidden_size, slot_size):
        super().__init__()
        self._obs_size = obs_size
        self._obs_channels = obs_channels
        self._decoder = nn.Sequential(
            Conv2dBlock(slot_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            conv2d(hidden_size, obs_channels + 1, 3, 1, 1),
        )
        self._pos_emb = PositionalEmbedding(obs_size, slot_size)

    def _spatial_broadcast(self, slots):
        slots = slots.unsqueeze(-1).unsqueeze(-1)
        return slots.repeat(1, 1, self._obs_size, self._obs_size)

    def forward(self, slots):
        B, N, _ = slots.shape
        # [batch_size * num_slots, d_slots]
        slots = slots.flatten(0, 1)
        # [batch_size * num_slots, d_slots, obs_size, obs_size]
        slots = self._spatial_broadcast(slots)
        out = self._decoder(self._pos_emb(slots))
        img_slots, masks = out[:, : self._obs_channels], out[:, -1:]
        img_slots = img_slots.view(
            B, N, self._obs_channels, self._obs_size, self._obs_size
        )
        masks = masks.view(B, N, 1, self._obs_size, self._obs_size)
        masks = masks.softmax(dim=1)
        recon_slots_masked = img_slots * masks
        return {"reconstruction": recon_slots_masked.sum(dim=1).flatten(start_dim=-2).movedim(2, 1),
                "masks": masks.flatten(start_dim=-3)}
