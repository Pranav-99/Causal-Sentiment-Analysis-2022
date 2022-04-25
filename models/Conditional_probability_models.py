import torch
from torch import nn
import torch.nn.functional as f

class TextFeaturizer(nn.Module):
    def __init__(self, embed_dim=128, gru_out=32):

        super(TextFeaturizer, self).__init__()
        self.embed_dim = embed_dim
        self.gru_out = gru_out
        self.vocab_size = 2000

        self.gru = nn.Sequential(
            nn.Embedding(self.vocab_size, embed_dim),
            nn.GRU(embed_dim, gru_out, bidirectional=True, batch_first=True),
        )

        self.output_dim = 2 * gru_out

    def forward(self, x):
        batch_size = x.size(0)
        x = self.gru(x)[1].transpose(0, 1).contiguous().view((x.shape[0], -1))
        assert(x.shape == (batch_size, self.output_dim))
        return x


class QPhi(nn.Module):
    def __init__(self, text_feat_dim=64, c_dim=2, z_dim=16, hidden_dim=32):

        super(QPhi, self).__init__()
        self.text_feat_dim = text_feat_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.y_dim = 1

        self.hidden_dim = hidden_dim

        self._model = nn.Sequential(
            nn.Linear(self.text_feat_dim + self.c_dim + self.y_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2 * self.z_dim)
        )

    def forward(self, x_feat, y, c):
        xcy = torch.cat([x_feat, c, y], dim=1)
        z_hat = self._model(xcy)
        z_hat_mu = z_hat[:, :z_hat.size(1) // 2]
        z_hat_logvar = z_hat[:, z_hat.size(1) // 2:]

        assert z_hat_mu.size(1) == z_hat_logvar.size(1) == self.z_dim
        return z_hat_mu, z_hat_logvar
    
    @staticmethod
    def sample(mu, logvar):
        # reparameterization
        std = torch.exp(logvar / 2.0)
        eps = torch.randn(std.shape, device=std.device)
        return mu + std * eps

class PThetaY_XZ(nn.Module):
    def __init__(self, text_feat_dim=64, z_dim=16):
        super(PThetaY_XZ, self).__init__()

        self.text_feat_dim = text_feat_dim
        self.z_dim = z_dim

        self.classifier = nn.Linear(text_feat_dim + z_dim, 1)

    def forward(self, x_feat, z):
        x = torch.cat((x_feat, z), dim=1)
        x = self.classifier(x)
        return x

class PThetaXfeat_Z(nn.Module):
    def __init__(self, z_dim=16, text_feat_dim=64, hidden_dim=32):
        super().__init__()

        self.z_dim = z_dim
        self.text_feat_dim = text_feat_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Linear(hidden_dim, text_feat_dim),
        )

    def forward(self, z):
        return self.model(z)


class PThetaC_Z(nn.Module):
    def __init__(self, z_dim=16, c_dim=2):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.projection = nn.Linear(z_dim, c_dim)

    def forward(self, z):
        return self.projection(z)



