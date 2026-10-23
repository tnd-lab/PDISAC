"""
RDPDNet -- hierarchical latent-variable RD-map denoiser (version label "pdnet"; earlier checkpoints carry "v4").

Probabilistic model (Sec. IV-B of the paper):
  encoder   q(z_l | z_{l-1}), l = 1..L, z_0 = Z_rd_prbs          (training only, Phi_enc)
  extractor p(z_L | Z_rd)                                        (highest-level prior, Phi_ext)
  decoder   p(z_l | z_{l+1}), l = 1..L-1, and p(Zhat | z_1)      (Phi_dec)
Z_rd enters only through the extractor prior p(z_L | Z_rd).

Design choices:
  (1) GroupNorm (per-sample normalisation, identical in train and eval).
  (2) ELBO-consistent top level: with teacher forcing every latent of the ELBO is drawn
      from q, so the decoder starts from z_L ~ q(z_L | z_{L-1}); the extractor prior
      enters only through D_KL( q(z_L | z_{L-1}) || p(z_L | Z_rd) ).
  (3) Linear output layers (GaussianHead and reconstruction).

Training path  f_den^train(Z_rd, Z_rd_prbs): forward(), returns Zhat and the L KL terms.
Inference path f_den^infer(Z_rd) = f_dec(f_ext(Z_rd)): predict(),
      z_L = mean of p(z_L | Z_rd);  z_l = mean of p(z_l | z_{l+1});  Zhat = mean of p(Zhat | z_1).
The hybrid objective (L_rec on f_den^train + L_rec_inf on f_den^infer) is formed in the
training scripts (lambda_inf).
"""

import torch
import torch.nn as nn

LOGV_MIN, LOGV_MAX = -10.0, 8.0


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def _norm(ch):
    """Per-sample normalisation (identical in train and eval)."""
    return nn.GroupNorm(num_groups=8 if ch % 8 == 0 else 1, num_channels=ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            _norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            _norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GaussianHead(nn.Module):
    """Linear head: feature map -> (mu, log-variance) of a diagonal Gaussian."""

    def __init__(self, in_ch, latent_ch):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 2 * latent_ch, kernel_size=3, padding=1)

    def forward(self, x):
        mu, logv = self.head(x).chunk(2, dim=1)
        return mu, torch.clamp(logv, LOGV_MIN, LOGV_MAX)


def reparameterize(mu, logv):
    """z = mu + sigma * eps,  eps ~ N(0, I)."""
    std = torch.exp(0.5 * torch.clamp(logv, LOGV_MIN, LOGV_MAX))
    return mu + std * torch.randn_like(std)


def kl_normal(mu_a, logv_a, mu_b, logv_b):
    """D_KL( N(mu_a, s_a^2) || N(mu_b, s_b^2) ), summed over latent elements, averaged over the batch."""
    logv_a = torch.clamp(logv_a, LOGV_MIN, LOGV_MAX)
    logv_b = torch.clamp(logv_b, LOGV_MIN, LOGV_MAX)
    va, vb = torch.exp(logv_a), torch.exp(logv_b) + 1e-8
    kl = 0.5 * (logv_b - logv_a + (va + (mu_a - mu_b).pow(2)) / vb - 1.0)
    return kl.flatten(1).sum(1).mean()


# --------------------------------------------------------------------------- #
#  Encoder  q(z_1, ..., z_L | Z_rd_prbs)   (training only)
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """z_0 := Z_rd_prbs,  q(z_l | z_{l-1}) = N(mu_{z_{l-1}}, Sigma_{z_{l-1}}),  l = 1..L."""

    def __init__(self, in_ch=2, hidden_channel=32, level=3):
        super().__init__()
        self.level = level
        self.feats = nn.ModuleList(
            [ConvBlock(in_ch if l == 0 else hidden_channel, hidden_channel) for l in range(level)]
        )
        self.q_heads = nn.ModuleList([GaussianHead(hidden_channel, hidden_channel) for _ in range(level)])

    def forward(self, Z_rd_prbs, sample=True):
        q_params, z_samples = [], []
        z_prev = Z_rd_prbs
        for l in range(self.level):
            mu, logv = self.q_heads[l](self.feats[l](z_prev))
            z_l = reparameterize(mu, logv) if sample else mu
            q_params.append((mu, logv))
            z_samples.append(z_l)
            z_prev = z_l
        return q_params, z_samples                      # index 0 -> level 1, index L-1 -> level L


# --------------------------------------------------------------------------- #
#  Extractor  p(z_L | Z_rd)
# --------------------------------------------------------------------------- #
class Extractor(nn.Module):
    def __init__(self, in_ch=2, hidden_channel=32):
        super().__init__()
        self.feat = nn.Sequential(ConvBlock(in_ch, hidden_channel), ConvBlock(hidden_channel, hidden_channel))
        self.p_head = GaussianHead(hidden_channel, hidden_channel)

    def forward(self, Z_rd):
        return self.p_head(self.feat(Z_rd))             # (mu, logv) of p(z_L | Z_rd)


# --------------------------------------------------------------------------- #
#  Decoder  p(Zhat | z_1) prod_{l=1}^{L-1} p(z_l | z_{l+1})
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    def __init__(self, hidden_channel=32, level=3, out_ch=2):
        super().__init__()
        self.level = level
        # index i handles the transition z_{L-i} -> z_{L-i-1}
        self.feats = nn.ModuleList([ConvBlock(hidden_channel, hidden_channel) for _ in range(level - 1)])
        self.p_heads = nn.ModuleList([GaussianHead(hidden_channel, hidden_channel) for _ in range(level - 1)])
        # mean of p(Zhat | z_1): linear output layer
        self.reconstruct = nn.Sequential(ConvBlock(hidden_channel, hidden_channel),
                                         nn.Conv2d(hidden_channel, out_ch, kernel_size=3, padding=1))

    def prior(self, i, z_upper):
        """p(z_{L-i-1} | z_{L-i}) parameters."""
        return self.p_heads[i](self.feats[i](z_upper))

    def forward(self, z_L, q_params=None, z_q=None, teacher_forcing=True, sample=True, distributions=None):
        """
        z_L      : top-level latent (from q in training with teacher forcing, from the extractor at inference)
        q_params : encoder posteriors [(mu, logv)]_{l=1..L}  (training only)
        z_q      : encoder posterior samples [z_1 .. z_L]     (training only)
        Returns Zhat and the lower-level KLs  D_KL( q(z_l | z_{l-1}) || p(z_l | z_{l+1}) ),  l = L-1 .. 1.
        """
        kls = []
        z_upper = z_L
        for i in range(self.level - 1):
            l = self.level - 1 - i                          # level being generated (L-1 .. 1), 1-based
            mu_p, logv_p = self.prior(i, z_upper)
            if distributions is not None:
                distributions[rf"$p(\vec{{\mathfrak{{z}}}}_{{{l}}}\mid\vec{{\mathfrak{{z}}}}_{{{l+1}}})$"] = mu_p
            if q_params is not None:                        # training
                mu_q, logv_q = q_params[l - 1]
                kls.append(kl_normal(mu_q, logv_q, mu_p, logv_p))
                z_upper = z_q[l - 1] if teacher_forcing else (reparameterize(mu_p, logv_p) if sample else mu_p)
            else:                                           # inference: prior chain
                z_upper = reparameterize(mu_p, logv_p) if sample else mu_p
        Zhat = self.reconstruct(z_upper)
        return Zhat, kls


# --------------------------------------------------------------------------- #
#  RDPDNet
# --------------------------------------------------------------------------- #
class RDPDNet(nn.Module):
    def __init__(
        self,
        input_shape=(2, 256, 256),
        hidden_channel=32,
        level=3,
        teacher_forcing=True,        # all ELBO latents drawn from q in training
        sample_at_inference=False,   # False: propagate means (deterministic denoiser)
    ):
        super().__init__()
        c, _, _ = input_shape
        self.level = level
        self.teacher_forcing = teacher_forcing
        self.sample_at_inference = sample_at_inference
        self.encoder = Encoder(in_ch=c, hidden_channel=hidden_channel, level=level)     # Phi_enc
        self.extractor = Extractor(in_ch=c, hidden_channel=hidden_channel)             # Phi_ext
        self.decoder = Decoder(hidden_channel=hidden_channel, level=level, out_ch=c)   # Phi_dec

    # f_den^train(Z_rd, Z_rd_prbs) = f_dec(f_ext(Z_rd), f_enc(Z_rd_prbs))
    def forward(self, Z_rd, Z_rd_prbs):
        q_params, z_q = self.encoder(Z_rd_prbs, sample=True)
        mu_ext, logv_ext = self.extractor(Z_rd)
        mu_qL, logv_qL = q_params[-1]
        kl_top = kl_normal(mu_qL, logv_qL, mu_ext, logv_ext)        # D_KL( q(z_L|z_{L-1}) || p(z_L|Z_rd) )
        z_L = z_q[-1] if self.teacher_forcing else reparameterize(mu_ext, logv_ext)
        Zhat, kls_lower = self.decoder(z_L, q_params, z_q, teacher_forcing=self.teacher_forcing)
        return Zhat, [kl_top] + kls_lower

    # f_den^infer(Z_rd) = f_dec(f_ext(Z_rd)); no torch.no_grad: the AFM steps backpropagate through it
    def predict(self, Z_rd):
        mu_ext, logv_ext = self.extractor(Z_rd)
        z_L = reparameterize(mu_ext, logv_ext) if self.sample_at_inference else mu_ext
        Zhat, _ = self.decoder(z_L, sample=self.sample_at_inference)
        return Zhat

    # training-path distributions (encoder chain, extractor prior, decoder priors)
    def distribution(self, Z_rd, Z_rd_prbs, distributions={}):
        q_params, z_q = self.encoder(Z_rd_prbs, sample=True)
        for l, (mu, _) in enumerate(q_params, start=1):
            distributions[rf"$q(\vec{{\mathfrak{{z}}}}_{{{l}}}\mid\vec{{\mathfrak{{z}}}}_{{{l-1}}})$"] = mu
        mu_ext, logv_ext = self.extractor(Z_rd)
        distributions[rf"$p(\vec{{\mathfrak{{z}}}}_{{{self.level}}}\mid\mathbf{{Z}}_{{\rm rd}})$"] = mu_ext
        mu_qL, logv_qL = q_params[-1]
        kl_top = kl_normal(mu_qL, logv_qL, mu_ext, logv_ext)
        z_L = z_q[-1] if self.teacher_forcing else reparameterize(mu_ext, logv_ext)
        Zhat, kls_lower = self.decoder(z_L, q_params, z_q, teacher_forcing=self.teacher_forcing,
                                       distributions=distributions)
        return Zhat, [kl_top] + kls_lower


if __name__ == "__main__":
    # smoke test: shapes, KL count (= L), train/eval consistency of predict
    c, hgt, wdt, L = 2, 64, 32, 3
    net = RDPDNet(input_shape=(c, hgt, wdt), hidden_channel=16, level=L)
    Y, Z = torch.randn(2, c, hgt, wdt), torch.randn(2, c, hgt, wdt)
    out, kls = net(Y, Z)
    assert out.shape == (2, c, hgt, wdt) and len(kls) == L, (out.shape, len(kls))
    p_train = net.predict(Y)
    net.eval()
    p_eval = net.predict(Y)
    assert torch.allclose(p_train, p_eval, atol=1e-5), "predict differs between train and eval mode"
    print("pdnet OK: output", tuple(out.shape), "| KL terms:", len(kls),
          "| params:", sum(p.numel() for p in net.parameters()))
