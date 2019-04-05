import torch
import torch.nn as nn
import torch.nn.functional as F

import math

NUMFREQ = 10
W_LATENT = 1024
Z_LATENT = 256

LOG_SQRT_2_PI = math.log(2*math.pi) / 2.0

VAR_OFFSET = 2.0*math.log(0.001)

def gaussian_kl(mu1, logvar1, mu2, logvar2):
    return 0.5*(
        logvar2 - logvar1
        + ((mu1 - mu2)**2 + torch.exp(logvar1)) * torch.exp(-logvar2)
        - 1.0
    )

def sample_gaussian(mu, logvar):
    eps = mu.new_empty(mu.size()).normal_()
    return mu + eps * torch.exp(logvar / 2.0)

class Residual(nn.Module):
    def __init__(self, N):
        super(Residual, self).__init__()
        self.fc_1 = nn.Linear(N, N)
        self.fc_2 = nn.Linear(N, N)

    def forward(self, x):
        return x + self.fc_2(F.elu(self.fc_1(x)))

class GraphBlock(nn.Module):
    def __init__(self, hin, hout):
        super(GraphBlock, self).__init__()
        # return f(h_i, sum_j!=i g(h_j))
        self.g_1 = Residual(hin)
        self.g_2 = Residual(hin)
        self.g_3 = Residual(hin)
        self.f_1 = nn.Linear(2*hin, hin+hout)
        self.f_2 = Residual(hin+hout)
        self.f_3 = nn.Linear(hin+hout, hout)

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
        h_f = F.elu(self.f_2(h_f))
        h_f = self.f_3(h_f)
        return h_f.view(bl, K, -1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # x preprocess
        # input is 200-long
        self.x_conv_1 = nn.Conv1d(1, 40, 10, 5) # now it is 39x40
        self.x_conv_2 = nn.Conv1d(40, 40, 7, padding=3)
        self.x_conv_3 = nn.Conv1d(40, 80, 6, 3) # now it is 12x80
        self.x_conv_4 = nn.Conv1d(80, 80, 7, padding=3)
        self.x_conv_5 = nn.Conv1d(80, 160, 4, 2) # now it is 5x160
        self.x_fc = Residual(800)
        # q(w_i | f_i, x, z)
        self.wi_embed_f = nn.Embedding(NUMFREQ, W_LATENT)
        self.wi_1 = GraphBlock(W_LATENT+Z_LATENT+800, 2048)
        self.wi_2 = GraphBlock(2048, 2048)
        self.wi_3 = GraphBlock(2048, 2048)
        #self.wi_4 = GraphBlock(2048, 2048)
        #self.wi_5 = GraphBlock(2048, 2048)
        self.wi_mu = nn.Linear(2048, W_LATENT)
        self.wi_logvar = nn.Linear(2048, W_LATENT)
        self.wi_rhos = nn.Linear(2048, W_LATENT)

        # q(z | x)
        self.z_fc_1 = nn.Linear(800, 2*Z_LATENT)
        self.z_fc_2 = Residual(2*Z_LATENT)
        self.z_fc_3 = Residual(2*Z_LATENT)
        #self.z_fc_4 = Residual(2*Z_LATENT)
        #self.z_fc_5 = Residual(2*Z_LATENT)
        self.z_fc_mu = nn.Linear(2*Z_LATENT, Z_LATENT)
        self.z_fc_logvar = nn.Linear(2*Z_LATENT, Z_LATENT)

    def preprocess_x(self, x):
        h = F.elu(self.x_conv_1(x.view(-1, 1, 200)))
        h = F.elu(self.x_conv_2(h))
        h = F.elu(self.x_conv_3(h))
        h = F.elu(self.x_conv_4(h))
        h = F.elu(self.x_conv_5(h))
        h = h.view(-1, 800)
        return F.elu(self.x_fc(h))

    def get_wis(self, hx, z, fs):
        batchlen = fs.size(0)
        K = fs.size(1)
        fis = self.wi_embed_f(fs.view(-1)).view(batchlen, K, W_LATENT)
        fis = fis + fis.new_empty(fis.size()).normal_()
        hx = hx.view(hx.size(0), 1, hx.size(1)).repeat(1, K, 1)
        z = z.view(z.size(0), 1, z.size(1)).repeat(1, K, 1)
        h = torch.cat((fis, hx, z), dim=2)
        h = F.elu(self.wi_1(h))
        h = F.elu(self.wi_2(h))
        h = F.elu(self.wi_3(h))
        #h = F.elu(self.wi_4(h))
        #h = F.elu(self.wi_5(h))
        h = h.view(-1, 2048)
        mus = self.wi_mu(h).view(batchlen, K, W_LATENT)
        logvars = self.wi_logvar(h).view(batchlen, K, W_LATENT)
        rhos = self.wi_rhos(h).view(batchlen, K, W_LATENT)
        return (mus, logvars, rhos)
    
    def get_z(self, hx):
        #h = self.preprocess_x(x)
        h = F.elu(self.z_fc_1(hx))
        h = F.elu(self.z_fc_2(h))
        h = F.elu(self.z_fc_3(h))
        #h = F.elu(self.z_fc_4(h))
        #h = F.elu(self.z_fc_5(h))
        mu = self.z_fc_mu(h)
        logvar = self.z_fc_logvar(h)
        return (mu, logvar)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # p(w_i | f_i)
        self.wi_fc_mu = nn.Embedding(NUMFREQ, W_LATENT)
        self.wi_fc_logvar = nn.Embedding(NUMFREQ, W_LATENT)
        # p(z | w)
        self.z_fc_1 = nn.Linear(W_LATENT, W_LATENT + Z_LATENT)
        #self.z_fc_2 = Residual(Z_LATENT+W_LATENT)
        #self.z_fc_3 = Residual(Z_LATENT+W_LATENT)
        self.z_fc_mu = nn.Linear(Z_LATENT+W_LATENT, Z_LATENT)
        self.z_fc_logvar = nn.Linear(Z_LATENT+W_LATENT, Z_LATENT)
        # p(x_t | w, z, x_t-1, ...)
        # - pre compute parameter
        self.x_fc_1 = nn.Linear(W_LATENT + Z_LATENT, 800)
        self.x_fc_2 = Residual(800)
        self.x_fc_3 = Residual(800)
        self.x_fc_4 = Residual(800)
        self.x_deconv_1 = nn.ConvTranspose1d(160, 80, 4, 2)
        self.x_conv_2 = nn.Conv1d(80, 80, 7, padding=3)
        self.x_deconv_3 = nn.ConvTranspose1d(80, 40, 8, 4)
        self.x_conv_4 = nn.Conv1d(40, 40, 7, padding=3)
        self.x_deconv_5 = nn.ConvTranspose1d(40, 20, 15, 5)
        self.x_conv_mu = nn.Conv1d(20, 1, 7, padding=3)
        self.x_conv_logvar = nn.Conv1d(20, 1, 7, padding=3)

    def get_wis(self, fs):
        K = fs.size(1)
        fis = fs.view(-1)
        mus = self.wi_fc_mu(fis).view(-1, K, W_LATENT)
        logvars = self.wi_fc_logvar(fis).view(-1, K, W_LATENT)
        logvars = F.elu(logvars - VAR_OFFSET - 1.0 ) + 1.0 + VAR_OFFSET
        return (mus, logvars)

    def get_z(self, w):
        h = F.elu(self.z_fc_1(w))
        #h = F.elu(self.z_fc_2(h))
        #h = F.elu(self.z_fc_3(h))
        mu = self.z_fc_mu(h)
        logvar = self.z_fc_logvar(h)
        logvar = F.elu(logvar - VAR_OFFSET - 1.0 ) + 1.0 + VAR_OFFSET
        return (mu, logvar)

    def get_x(self, w, z, logvar=None):
        batchlen = w.size(0)
        h = F.elu(self.x_fc_1(torch.cat((w, z), dim=1)))
        h = F.elu(self.x_fc_2(h))
        h = F.elu(self.x_fc_3(h))
        h = F.elu(self.x_fc_4(h))
        h = h.view(-1,160,5)
        h = F.elu(self.x_deconv_1(h))
        h = F.elu(self.x_conv_2(h))
        h = F.elu(self.x_deconv_3(h))
        h = F.elu(self.x_conv_4(h))
        h = F.elu(self.x_deconv_5(h))
        mu = self.x_conv_mu(h)[:,:,35:-35]
        if logvar is None:
            logvar = self.x_conv_logvar(h)[:,:,35:-35]
        else:
            logvar = w.new_zeros((batchlen, 200)) + logvar
        # constraint minimum variance: sigma=0.001
        logvar = F.elu(logvar - VAR_OFFSET - 1.0 ) + 1.0 + VAR_OFFSET
        return (mu.view(-1, 200), logvar.view(-1,200))

    def nll_x(self, x, w, z, logvar=None):
        (mu, logvar) = self.get_x(w, z, logvar=logvar)
        return 0.5*torch.sum(torch.exp(-logvar) * (mu - x) ** 2 + logvar - VAR_OFFSET) / x.size(0)

    def sample_x(self, w, z, logvar=None):
        batchlen = w.size(0)
        (mu, logvar) = self.get_x(w, z, logvar=logvar)
        eps = w.new_zeros((batchlen, 200)).normal_(0,1)
        return mu + torch.exp(logvar/2) * eps


class VAE(nn.Module):
    def __init__(self, x_logvar=None):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.generator = Generator()
        self.x_logvar = x_logvar
        # contract the initial weights for better stability
        #for p in self.parameters():
        #    p.data *= 0.8

    def rec_losses(self, x, fs):
        batchlen = fs.size(0)
        K = fs.size(1)
        hx = self.encoder.preprocess_x(x)
        (z_mu_q, z_logvar_q) = self.encoder.get_z(hx)
        z = sample_gaussian(z_mu_q, z_logvar_q)
        # compute the wis and their loss
        (wi_mus_q, wi_logvars_q, wi_rhos_q) = self.encoder.get_wis(hx, z, fs-1)
        (wi_mus_p, wi_logvars_p) = self.generator.get_wis(fs-1)
        logrhos = F.log_softmax(torch.cat((wi_rhos_q, hx.new_zeros((batchlen, 1, W_LATENT))), dim=1), dim=1)
        rhos = torch.exp(logrhos[:,:K,:])
        logonemsumrhos = logrhos[:,K,:]
        eps = x.new_empty(wi_mus_q.size()).normal_()
        wis = torch.exp(wi_logvars_q/2.0) * (eps - rhos * torch.sum(eps, dim=1, keepdim=True)) + wi_mus_q
        loss_w = 0.5 * torch.sum(
            torch.sum(wi_logvars_p - wi_logvars_q, dim=1) - 2 * logonemsumrhos
            + torch.sum((wi_mus_q-wi_mus_p)**2 * torch.exp(-wi_logvars_p), dim=1)
            + torch.sum((1.0 - 2*rhos + K * rhos**2) * torch.exp(wi_logvars_q-wi_logvars_p), dim=1)
            - K
        ) / x.size(0)
        w = torch.sum(wis, dim=1)
        # compute the z loss
        (z_mu_p, z_logvar_p) = self.generator.get_z(w)
        loss_z = 0.5*torch.sum((z-z_mu_p)**2 * torch.exp(-z_logvar_p) + z_logvar_p - z_logvar_q -1) / z.size(0)
        loss_x = self.generator.nll_x(x, w, z, logvar=self.x_logvar)
        return (loss_w, loss_z, loss_x, K)

    def rec_output(self, x, fs):
        batchlen = fs.size(0)
        K = fs.size(1)
        hx = self.encoder.preprocess_x(x)
        (z_mu_q, z_logvar_q) = self.encoder.get_z(hx)
        z = sample_gaussian(z_mu_q, z_logvar_q)
        (wi_mus_q, wi_logvars_q, wi_rhos_q) = self.encoder.get_wis(hx, z, fs-1)
        logrhos = F.log_softmax(torch.cat((wi_rhos_q, hx.new_zeros((batchlen, 1, W_LATENT))), dim=1), dim=1)
        rhos = torch.exp(logrhos[:,:K,:])
        eps = x.new_empty(wi_mus_q.size()).normal_()
        wis =torch.exp(wi_logvars_q/2.0) * (eps - rhos * torch.sum(eps, dim=1, keepdim=True)) + wi_mus_q
        w = torch.sum(wis, dim=1)
        (mu, logvar) = self.generator.get_x(w, z, logvar=self.x_logvar)
        return (mu, torch.exp(0.5 * logvar))

    def generate(self, fs):
        (wi_mus_p, wi_logvars_p) = self.generator.get_wis(fs-1)
        wis = sample_gaussian(wi_mus_p, wi_logvars_p)
        w = torch.sum(wis, dim=1)
        (z_mu_p, z_logvar_p) = self.generator.get_z(w)
        z = sample_gaussian(z_mu_p, z_logvar_p)
        return self.generator.sample_x(w, z, logvar=self.x_logvar)

    def generate_same_latent(self, fs, samples):
        assert fs.size(0) == 1
        (wi_mus_p, wi_logvars_p) = self.generator.get_wis(fs-1)
        wis = sample_gaussian(wi_mus_p, wi_logvars_p)
        w = torch.sum(wis, dim=1)
        # tile the latent variables to only vary on sampling x and z
        w = w.repeat(samples, 1)
        (z_mu_p, z_logvar_p) = self.generator.get_z(w)
        z = sample_gaussian(z_mu_p, z_logvar_p)
        return self.generator.sample_x(w, z, logvar=self.x_logvar)
