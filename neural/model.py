import math
import torch
import torch.nn.functional as F
from torch import nn


class PlateNet(nn.Module):
    def __init__(self, n_modes=32, latent_dim=128, fourier_bands=32):
        super().__init__()
        self.fourier_bands = fourier_bands

        self.geometry_net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, latent_dim), nn.SiLU(),
        )
        self.frequency_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, n_modes),
        )
        self.point_head = nn.Sequential(
            nn.Linear(latent_dim + 2 + 4 * fourier_bands, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, n_modes),
        )

        output = self.frequency_head[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            output.bias.zero_()
            output.bias[0] = math.log(20.0)
            output.bias[1:] = -3.2

    def encode_geometry(self, geometry):
        return self.geometry_net(torch.stack(
            (2 * geometry[:, 0] - 1, (geometry[:, 1] - 1.25) / 0.75), dim=1
        ))

    def predict_log_factors(self, z):
        raw = self.frequency_head(z)
        first = raw[:, :1]
        return torch.cat((first, first + torch.cumsum(F.softplus(raw[:, 1:]) + 1e-5, dim=1)), dim=1)

    def point_features(self, points):
        features = [points]
        x, y = points[..., :1], points[..., 1:2]
        for k in range(1, self.fourier_bands + 1):
            a = k * math.pi
            features += [torch.sin(a * x), torch.cos(a * x), torch.sin(a * y), torch.cos(a * y)]
        return torch.cat(features, dim=-1)

    def predict_gains(self, z, points):
        z = z[:, None, :].expand(-1, points.shape[1], -1)
        return self.point_head(torch.cat((z, self.point_features(points)), dim=-1))

    def forward(self, geometry, points):
        z = self.encode_geometry(geometry)
        return torch.exp(self.predict_log_factors(z)), self.predict_gains(z, points)
