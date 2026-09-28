import math
import torch
import torch.nn.functional as F
from torch import nn

# Neural network for predicting geometry-dependent frequency factors and gains at specific plate positions
# learns from FEM reference data to replace expensive FEM solving

class PlateNet(nn.Module):
    # sets up network with geometry encoder, frequency and point gain heads
    def __init__(self, n_modes=32, latent_dim=128, fourier_bands=32):
        super().__init__()
        self.fourier_bands = fourier_bands

        # takes morph & aspect and encodes into geometry representation
        self.geometry_net = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(),
            nn.Linear(128, latent_dim), nn.SiLU(),
        )

        # predicts mode frequency factors from geometry representation 
        self.frequency_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, n_modes),
        )

        # predicts mode gains at sample points 
        self.point_head = nn.Sequential(
            nn.Linear(latent_dim + 2 + 4 * fourier_bands, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, n_modes),
        )

        # init frequency head starting values 
        output = self.frequency_head[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            output.bias.zero_()
            output.bias[0] = math.log(20.0)
            output.bias[1:] = -3.2

    # normalizes geometry parameters and encode into geometry vector/representation
    def encode_geometry(self, geometry):
        return self.geometry_net(torch.stack(
            (2 * geometry[:, 0] - 1, (geometry[:, 1] - 1.25) / 0.75), dim=1
        ))

    # predicts frequency factors from geometry vector and orders them ascending
    def predict_log_factors(self, z):
        raw = self.frequency_head(z)
        first = raw[:, :1]
        return torch.cat((first, first + torch.cumsum(F.softplus(raw[:, 1:]) + 1e-5, dim=1)), dim=1)

    # encode positions of sample point coordinates with fourier features instead of giving raw coordinates, 
    # because NN struggled to learn with higher plate modes as they oscillate rapidly
    def point_features(self, points):
        features = [points]
        x, y = points[..., :1], points[..., 1:2]
        for k in range(1, self.fourier_bands + 1):
            a = k * math.pi
            features += [torch.sin(a * x), torch.cos(a * x), torch.sin(a * y), torch.cos(a * y)]
        return torch.cat(features, dim=-1)

    # predict gains at sample points using geometry vector and point coordinates
    def predict_gains(self, z, points):
        z = z[:, None, :].expand(-1, points.shape[1], -1)
        return self.point_head(torch.cat((z, self.point_features(points)), dim=-1))

    # run network: takes geometry and points, returns frequency factors and gains at points
    def forward(self, geometry, points):
        z = self.encode_geometry(geometry)
        return torch.exp(self.predict_log_factors(z)), self.predict_gains(z, points)
