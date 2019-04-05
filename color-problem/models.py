import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

NUM_COLORS = 5
A_EMBED = 512
W_EMBED = 2048
Z_LATENT = 1024
L_EMBED = 32

def sample_gaussian(mu, logvar):
    eps = mu.new_empty(mu.size()).normal_()
    return mu + eps * torch.exp(logvar/2)

def gaussian_kl(mu1, logvar1, mu2, logvar2):
    return 0.5 * (
        logvar2 - logvar1
        + (mu1 - mu2)**2 * torch.exp(-logvar2) 
        + torch.exp(logvar1 - logvar2)
        - 1.0
    )

def discretized_logistic(x, mu, s):
    ones = (x > (254.5/255.0)).float()
    zeros = (x < (0.5/255.0)).float()
    return torch.sum(
        F.softplus((x - 0.5/255.0 - mu) / s) * (1.0 - zeros)
        + F.softplus(- (x + 0.5/255.0 - mu) / s) * (1.0 - ones)
         - torch.log(1.0 - torch.exp(-1.0 / (255.0 * s))) * (1.0 - zeros) * (1.0 - ones)
    ) / x.size(0)


P_VAR_SCALE = 1
P_VAR_OFFSET = 2.0*math.log(0.005)
X_VAR_OFFSET = 2.0*math.log(0.005)


def output_nll(x, mu, logvar):
    return 0.5*torch.sum(((x-mu)**2)*torch.exp(-logvar) + logvar - X_VAR_OFFSET)/x.size(0)
    #return discretized_logistic(x, mu, F.softplus(logvar/2.0))

class Residual(nn.Module):
    def __init__(self, N):
        super(Residual, self).__init__()
        self.fc_1 = nn.Linear(N, N)
        self.fc_2 = nn.Linear(N, N)

    def __call__(self, x):
        return x + self.fc_2(F.elu(self.fc_1(x)))

class GraphBlock(nn.Module):
    def __init__(self, hin, hout):
        super(GraphBlock, self).__init__()
        # return f(h_i, sum_j!=i g(h_j))
        self.g_1 = Residual(hin)
        self.g_2 = Residual(hin)
        self.g_3 = Residual(hin)
        self.f_1 = nn.Linear(2*hin, hout)
        self.f_2 = Residual(hout)

    def forward(self, h):
        # h must be of size (batchlen, K, hin)
        bl = h.size(0)
        K = h.size(1)
        # compute sum_{i!=j} g(h_j)
        h = h.view(bl*K, -1)
        h_g = F.elu(self.g_1(h)).view(bl, K, -1)
        h_g = torch.sum(h_g, dim=1, keepdim=True) - h_g
        h_g = h_g.view(bl*K, -1)
        h_g = torch.tanh(self.g_2(h_g))
        h_g = self.g_3(h_g)
        # compute f
        h_f = torch.cat((h, h_g), dim=1)
        h_f = F.elu(self.f_1(h_f))
        h_f = self.f_2(h_f)
        return h_f.view(bl, K, -1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.x_1 = nn.Conv2d( 3, 16, 5, 1, 2)
        self.x_2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.x_3 = nn.Conv2d(32, 32, 5, 1, 2)
        self.x_4 = nn.Conv2d(32, 48, 4, 2, 1)
        self.x_5 = nn.Conv2d(48, 64, 4, 2, 1)
        self.x_8 = Residual(1024)
        # q(z | x)
        self.z_1 = Residual(1024)
        #self.z_2 = Residual(1024)
        #self.z_3 = Residual(1024)
        self.z_mu = nn.Linear(1024, 1024)
        self.z_logvar = nn.Linear(1024, 1024)
        # q(wi | x, li, z)
        self.li_in_col = nn.Embedding(NUM_COLORS, L_EMBED)
        self.li_in_loc = nn.Linear(2, L_EMBED)
        self.w_1 = GraphBlock(1024+2*L_EMBED+Z_LATENT, 2048)
        self.w_2 = GraphBlock(2048, 2048)
        self.w_3 = GraphBlock(2048, 2048)
        #self.w_4 = GraphBlock(2048, 2048)
        #self.w_5 = GraphBlock(2048, 2048)
        self.w_mus = nn.Linear(2048, 2048)
        self.w_logvars = nn.Linear(2048, 2048)
        self.w_rhos = nn.Linear(2048, 2048)
        # li embedding
   
    def process_x(self, x):
        h = F.elu(self.x_1(x))
        h = F.elu(self.x_2(h))
        h = F.elu(self.x_3(h))
        h = F.elu(self.x_4(h))
        h = F.elu(self.x_5(h))
        h = h.view(-1, 1024)
        h = F.elu(self.x_8(h))
        return h

    def get_z(self, h_x):
        h = F.elu(self.z_1(h_x))
        #h = F.elu(self.z_2(h))
        #h = F.elu(self.z_3(h))
        mu = self.z_mu(h)
        logvar = self.z_logvar(h)
        return (mu, logvar)

    def get_wis(self, ls, h_x, z):
        (ls_col, ls_loc) = ls
        batchlen = ls_col.size(0)
        K = ls_col.size(1)
        h_l = torch.cat((
                self.li_in_col(ls_col.view(-1)).view(batchlen, K, L_EMBED),
                self.li_in_loc(ls_loc.view(-1, 2)).view(batchlen, K, L_EMBED),
            ),
            dim=2
        )
        h_zx = torch.cat((h_x, z), dim=1)
        h_zx = h_zx.view(batchlen, 1, 1024+Z_LATENT).repeat(1, K, 1)
        #h_x = h_x.view(batchlen, 1, 1024).repeat(1, K, 1)
        h = torch.cat((h_l, h_zx), dim=2)
        h = F.elu(self.w_1(h))
        h = F.elu(self.w_2(h))
        h = F.elu(self.w_3(h))
        #h = F.elu(self.w_4(h))
        #h = F.elu(self.w_5(h))
        h = h.view(-1, 2048)
        mus = self.w_mus(h).view(-1, K, W_EMBED)
        logvars = self.w_logvars(h).view(-1, K, W_EMBED)
        rhos = self.w_rhos(h).view(-1, K, W_EMBED)
        return (mus, logvars, rhos)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # p(w_i | l_i)
        self.li_in_col = nn.Embedding(NUM_COLORS, L_EMBED)
        self.li_in_loc = nn.Linear(2, L_EMBED)
        self.w_1 = nn.Linear(2*L_EMBED, 1024)
        #self.w_2 = Residual(1024)
        #self.w_3 = Residual(1024)
        self.w_mu = nn.Linear(1024, W_EMBED)
        self.w_logvar = nn.Linear(1024, W_EMBED)
        # p(z | w)
        self.z_1 = nn.Linear(2048, 1024)
        self.z_2 = Residual(1024)
        #self.z_3 = Residual(1024)
        #self.z_4 = Residual(1024)
        self.z_mu = nn.Linear(1024, 1024)
        self.z_logvar = nn.Linear(1024, 1024)
        # p(x | z, w)
        self.x_z_in = nn.Linear(1024, 2048)
        self.x_1 = Residual(2048)
        self.x_2 = Residual(2048)
        self.x_3 = Residual(2048)
        #self.x_4 = Residual(2048)
        #self.x_5 = nn.Conv2d(128, 96, 5, 1, 2)
        self.x_6 = nn.Conv2d(128, 64, 5, 1, 2)
        self.x_7 = nn.Conv2d(64, 32, 5, 1, 2)
        self.x_8 = nn.Conv2d(32, 24, 5, 1, 2)
        self.x_mu = nn.Conv2d(24, 3, 5, 1, 2)
        self.x_logvar = nn.Conv2d(24, 1, 5, 1, 2)
    
    def get_wis(self, ls):
        (ls_col, ls_loc) = ls
        batchlen = ls_col.size(0)
        K = ls_col.size(1)
        h = torch.cat((
                self.li_in_col(ls_col.view(-1)),
                F.elu(self.li_in_loc(ls_loc.view(-1, 2)))
            ),
            dim=1
        )
        h = F.elu(self.w_1(h))
        #h = F.elu(self.w_2(h))
        #h = F.elu(self.w_3(h))
        mus = self.w_mu(h).view(-1, K, W_EMBED)
        logvars = self.w_logvar(h).view(-1, K, W_EMBED) / P_VAR_SCALE
        logvars = F.elu(logvars - P_VAR_OFFSET - 1.0 ) + 1.0 + P_VAR_OFFSET
        return (mus, logvars)
    
    def get_z(self, w):
        h = F.elu(self.z_1(w))
        h = F.elu(self.z_2(h))
        #h = torch.tanh(self.z_3(h))
        #h = F.elu(self.z_4(h))
        mu = self.z_mu(h)
        logvar = self.z_logvar(h) / P_VAR_SCALE
        logvar = F.elu(logvar - P_VAR_OFFSET - 1.0 ) + 1.0 + P_VAR_OFFSET
        return (mu, logvar)
    
    def get_x(self, w, z):
        h = F.elu(self.x_1(w) + self.x_z_in(z))
        h = torch.tanh(self.x_2(h))
        h = F.elu(self.x_3(h))
        #h = F.elu(self.x_4(h))
        h = h.view(-1, 128, 4, 4)
        #h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        #h = F.elu(self.x_5(h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        h = F.elu(self.x_6(h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        h = F.elu(self.x_7(h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=True)
        h = F.elu(self.x_8(h))
        mu = self.x_mu(h)
        logvar = self.x_logvar(h)
        logvar = F.elu(logvar - X_VAR_OFFSET - 1.0 ) + 1.0 + X_VAR_OFFSET
        return (mu, logvar)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.generator = Generator()

    def rec_with_losses(self, x, labels):
        batchlen = labels[0].shape[0]
        K = labels[1].shape[1]
        hx = self.encoder.process_x(x*2.0 - 1.0)
        (z_mu_q, z_logvar_q) = self.encoder.get_z(hx)
        z = sample_gaussian(z_mu_q, z_logvar_q)
        # compute the wis
        (wi_mus_q, wi_logvars_q, wi_rhos_q) = self.encoder.get_wis(labels, hx, z)
        logrhos = F.log_softmax(torch.cat((wi_rhos_q, hx.new_zeros((batchlen, 1, W_EMBED))), dim=1), dim=1)
        rhos = torch.exp(logrhos[:,:K,:])
        logonemsumrhos = logrhos[:,K,:]
        eps = x.new_empty(wi_mus_q.size()).normal_()
        wis = torch.exp(wi_logvars_q/2.0) * (eps - rhos * torch.sum(eps, dim=1, keepdim=True)) + wi_mus_q
        (wi_mus_p, wi_logvars_p) = self.generator.get_wis(labels)
        loss_w = 0.5 * torch.sum(
                torch.sum(wi_logvars_p - wi_logvars_q, dim=1) - 2 * logonemsumrhos
                + torch.sum((wi_mus_q-wi_mus_p)**2 * torch.exp(-wi_logvars_p), dim=1)
                + torch.sum((1.0 - 2*rhos + K * rhos**2) * torch.exp(wi_logvars_q-wi_logvars_p), dim=1)
                - K
        ) / x.size(0)
        w = torch.sum(wis, dim=1)
        # compute the z loss
        (z_mu_p, z_logvar_p) = self.generator.get_z(w)
        loss_z = 0.5*torch.sum((z-z_mu_p)**2 * torch.exp(-z_logvar_p) + z_logvar_p - z_logvar_q - 1) / z.size(0)
        (x_mu, x_logvar) = self.generator.get_x(w, z)
        loss_x = output_nll(x, x_mu, x_logvar)
        return (torch.clamp(x_mu, 0.0, 1.0), loss_w, loss_z, loss_x, K)

    def generate(self, labels):
        (wi_mu_p, wi_logvar_p) = self.generator.get_wis(labels)
        wis = sample_gaussian(wi_mu_p, wi_logvar_p)
        w = torch.sum(wis, dim=1)
        z = sample_gaussian(*self.generator.get_z(w))
        (x_mu, _) = self.generator.get_x(w, z)
        return torch.clamp(x_mu, 0.0, 1.0)
