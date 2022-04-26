import torch
from torch import nn
import torch.nn.functional as F

from models.Conditional_probability_models import TextFeaturizer, QPhi, PThetaY_XZ, PThetaXfeat_Z, PThetaC_Z
from utils import recon_loss_function, kl_loss_function

class StructuralCausalModel(nn.Module):
    def __init__(
        self,
        text_embed_dim=128,
        gru_out=32,
        c_dim=2,
        z_dim=16,
        q_phi_hidden_dim=32,
        pthetaxfeat_z_hidden_dim=32,
        lambda_recon=1.,
        labmda_y_xz=1.,
        lambda_c_z=1.,
        lambda_KL=1.,
    ):
        super().__init__()

        self.text_embed_dim = text_embed_dim
        self.gru_out = gru_out
        self.text_feat_dim = 2 * gru_out
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.q_phi_hidden_dim = q_phi_hidden_dim
        self.pthetaxfeat_z_hidden_dim = pthetaxfeat_z_hidden_dim
        self.lambda_recon = lambda_recon
        self.labmda_y_xz = labmda_y_xz
        self.lambda_c_z = lambda_c_z
        self.lambda_KL = lambda_KL

        self.text_featurizer = TextFeaturizer(embed_dim=text_embed_dim, gru_out=gru_out)
        self.q_phi = QPhi(text_feat_dim=self.text_feat_dim, c_dim=c_dim, z_dim=z_dim, hidden_dim=q_phi_hidden_dim)

        self.pthetay_xz = PThetaY_XZ(text_feat_dim=self.text_feat_dim, z_dim=z_dim)

        self.pthetaxfeat_z = PThetaXfeat_Z(z_dim=z_dim, text_feat_dim=self.text_feat_dim, hidden_dim=pthetaxfeat_z_hidden_dim)
        self.pthetac_z = PThetaC_Z(z_dim=z_dim, c_dim=c_dim)

        self.cross_entropy_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, c, return_all_losses=False):
        ''' returns losses '''
        x_feat = self.text_featurizer(x)
        # print("x.shape", x.shape)
        # print("y.shape", y.shape)
        # print("c.shape", c.shape)
        # print("x_feat.shape", x_feat.shape)

        z_hat_mu, z_hat_logvar = self.q_phi(x_feat, y, c)
        # print("z_hat_mu.shape", z_hat_mu.shape)
        # print("z_hat_logvar.shape", z_hat_logvar.shape)

        ## TODO: Consider running with more z's per sample
        z = self.q_phi.sample(z_hat_mu, z_hat_logvar)
        # print("z.shape", z.shape)

        y_hat_logits = self.pthetay_xz(x_feat, z)
        # print("y_hat_logits.shape", y_hat_logits.shape)
        x_feat_hat = self.pthetaxfeat_z(z)
        # print("x_feat_hat.shape", x_feat_hat.shape)
        c_hat_logits = self.pthetac_z(z)
        # print("c_hat_logits.shape", c_hat_logits.shape)

        L_recon = recon_loss_function(x_feat, x_feat_hat)
        L_y_xz = self.cross_entropy_loss(y_hat_logits, y).squeeze(dim=1) # cross entropy
        L_c_z = self.cross_entropy_loss(c_hat_logits, c).sum(dim=1) # cross entropy

        L_KL = kl_loss_function(z_hat_mu, z_hat_logvar)

        # print("L_recon.shape", L_recon.shape)
        # print("L_y_xz.shape", L_y_xz.shape)
        # print("L_c_z.shape", L_c_z.shape)
        # print("L_KL.shape", L_KL.shape)

        assert L_recon.requires_grad
        assert L_y_xz.requires_grad
        assert L_c_z.requires_grad
        assert L_KL.requires_grad

        loss = self.lambda_recon * L_recon \
                 + self.labmda_y_xz * L_y_xz \
                 + self.lambda_c_z * L_c_z \
                 + self.lambda_KL * L_KL

        if return_all_losses:
            return loss.mean(), L_recon.mean(), L_y_xz.mean(), L_c_z.mean(), L_KL.mean()
        else:
            return loss.mean()

    def forward_evaluation(self, x, y, c, return_all_losses=False):
        ''' returns losses '''
        self.eval()
        with torch.no_grad():
            x_feat = self.text_featurizer(x)

            z_hat_mu, z_hat_logvar = self.q_phi(x_feat, y, c)

            ## TODO: Consider running with more z's per sample
            z = self.q_phi.sample(z_hat_mu, z_hat_logvar)

            y_hat_logits = self.pthetay_xz(x_feat, z)
            x_feat_hat = self.pthetaxfeat_z(z)
            c_hat_logits = self.pthetac_z(z)

            L_recon = recon_loss_function(x_feat, x_feat_hat)
            L_y_xz = self.cross_entropy_loss(y_hat_logits, y).squeeze(dim=1) # cross entropy
            L_c_z = self.cross_entropy_loss(c_hat_logits, c).sum(dim=1) # cross entropy

            L_KL = kl_loss_function(z_hat_mu, z_hat_logvar)

            loss = self.lambda_recon * L_recon \
                    + self.labmda_y_xz * L_y_xz \
                    + self.lambda_c_z * L_c_z \
                    + self.lambda_KL * L_KL

        self.train()

        if return_all_losses:
            return loss.mean(), L_recon.mean(), L_y_xz.mean(), L_c_z.mean(), L_KL.mean()
        else:
            return loss.mean()


    ## TODO: Move into SCM class
    def generate_z(self, x, y, c):
        x_feat = self.text_featurizer(x)
        z_hat_mu, z_hat_logvar = self.q_phi(x_feat, y, c)
        z_hat = self.q_phi.sample(z_hat_mu, z_hat_logvar)
        return z_hat

    ## TODO: Move into StructuralCausalModel
    def predict(self, x, gan, Z_SAMPLE_COUNT=32, device='cuda'):
        x_dup = torch.repeat_interleave(x, Z_SAMPLE_COUNT, dim=0)
        z_samples = gan.drawsamples(N=len(x_dup), get_tensor=True)

        # x_dup = x_dup.to(device)
        # z_samples = z_samples.to(device)

        with torch.no_grad():
            x_feat = self.text_featurizer(x_dup)
            y_pred_logits = self.pthetay_xz(x_feat, z_samples).reshape((x.size(0), Z_SAMPLE_COUNT, -1)).mean(dim=1)

        return nn.functional.sigmoid(y_pred_logits)
