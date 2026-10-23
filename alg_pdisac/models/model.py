import time
import torch
from torch import nn

from models.bases import dncnn, pdnet


class DenoiseModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super(DenoiseModel, self).__init__()
        self.model = model

    def convert_to_complex(self, x):
        return torch.complex(x[:, 0, ...], x[:, 1, ...])

    def convert_from_complex(self, x):
        real_part = x.real
        imag_part = x.imag
        return torch.stack([real_part, imag_part], dim=1)

    def forward(self, Z_rd: torch.Tensor = None, Z_rd_prbs: torch.Tensor = None):
        if isinstance(Z_rd_prbs, torch.Tensor):
            output, kls = self.model(Z_rd=Z_rd, Z_rd_prbs=Z_rd_prbs)
            return output, kls
        else:
            output = self.model(Z_rd=Z_rd)
            return output

    def predict(self, Z_rd: torch.Tensor = None):
        output = self.model.predict(Z_rd=Z_rd)
        return output


    def create_mask(self, Z_rd_prbs):
        # Z_rd_prbs: (B, 2, H, W) -> channel 0 = real, channel 1 = imag

        # Magnitude of the complex value at each pixel
        mag = torch.sqrt(Z_rd_prbs[:, 0] ** 2 + Z_rd_prbs[:, 1] ** 2 + 1e-12)  # (B, H, W)

        # Normalize per sample in batch
        peak_ref = mag.flatten(1).max(dim=1).values.view(-1, 1, 1)  # (B, 1, 1)
        M = mag / (peak_ref + 1e-12)
        M = torch.clamp(M, 0, 1)

        # Sharpen: suppress weak clutter, highlight target peak
        M = M ** 2  # (B, H, W)

        # Same mask applied to both real & imag channels
        M_prbs = M.unsqueeze(1).expand(-1, 2, -1, -1)  # (B, 2, H, W)
        return M_prbs

    def distribution(
        self,
        Z_rd: torch.Tensor = None,
        Z_rd_prbs: torch.Tensor = None,
        distributions: dict = {},
    ):
        distributions[r"$p(\mathbf{Z}_{\rm rd})$"] = Z_rd
        distributions[r"$p(\mathbf{Z}_{\rm rd}^{\rm prbs})$"] = Z_rd_prbs

        Z_rd_hat, kls = self.model.distribution(
            Z_rd=Z_rd, Z_rd_prbs=Z_rd_prbs, distributions=distributions
        )

        distributions[r"$p(\mathbf{\hat{Z}}_{\rm rd})$"] = Z_rd_hat

        return distributions


def count_parameters(model, inputs=None):
    """Number of trainable parameters.

    inputs = None : all parameters of the model (RDPDNet: encoder + extractor + decoder).
    inputs = dict : only the parameters of the modules executed by model.predict(**inputs),
                    i.e. the run-time network (RDPDNet: extractor + decoder; the encoder is
                    used in training only). Shared parameters are counted once.
    """
    if inputs is None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    used, handles = set(), []
    for m in model.modules():
        handles.append(m.register_forward_hook(lambda mod, inp, out: used.add(mod)))
    model.eval()
    with torch.no_grad():
        model.predict(**inputs)
    for hdl in handles:
        hdl.remove()
    params = {id(p): p for m in used for p in m.parameters(recurse=False) if p.requires_grad}
    return sum(p.numel() for p in params.values())


def synchronize(device):
    """Wait for the queued kernels of an accelerator (asynchronous on CUDA and MPS)."""
    dev = torch.device(device).type
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def measure_inference_time(model, inputs, num_runs=100, warmup=10, device="cpu"):
    """Median wall time of one model.predict(**inputs) call [ms] on `device`.

    The batch is whatever `inputs` holds (time per frame = result / batch size).
    The device is synchronized before and after every timed call, since CUDA and MPS
    kernels run asynchronously.
    """
    model.eval()
    model.to(device)
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model.predict(**inputs)
        for _ in range(num_runs):
            synchronize(device)
            t0 = time.perf_counter()
            model.predict(**inputs)
            synchronize(device)
            times.append(time.perf_counter() - t0)
    times.sort()
    return 1000 * times[len(times) // 2]


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def count_flops(model, inputs):
    """Count the forward-pass MACs/FLOPs of one frame (batch = 1) via hooks.

    Convention: 1 MAC = 1 multiply + 1 add, so FLOPs = 2 x MACs, the standard
    used in the complexity table. Only Conv2d, ConvTranspose2d, and
    Linear layers are counted; they dominate the cost, while BatchNorm and
    activations are negligible. All layers here use stride 1, so the
    output-based MAC formula is exact.

    Returns
    -------
    (macs, flops) : tuple of int
        Total multiply-accumulate operations and FLOPs (= 2 * macs) per frame.
    """
    macs = 0
    handles = []

    def conv_hook(module, inp, out):
        nonlocal macs
        out_c = out.shape[1]
        out_hw = out.shape[2] * out.shape[3]
        in_c = inp[0].shape[1]
        ksize = module.kernel_size
        kh, kw = ksize if isinstance(ksize, tuple) else (ksize, ksize)
        macs += out_c * out_hw * (in_c // module.groups) * kh * kw

    def linear_hook(module, inp, out):
        nonlocal macs
        macs += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        _ = model.predict(**inputs)

    for hdl in handles:
        hdl.remove()

    return macs, 2 * macs


if __name__ == "__main__":
    device = "cuda"
    print(f"Using device: {device}\n")

    c, h, w = (2, 512, 256)

    results = []

    print("=" * 80)
    print("TESTING AND BENCHMARKING MODELS")
    print("=" * 80)

    print("\n1. Testing RDPDNet ...")
    Y = torch.randn(1, c, h, w)
    Z = torch.randn(1, c, h, w)

    current_model = pdnet.RDPDNet(input_shape=(c, h, w), hidden_channel=16, level=3)
    model = DenoiseModel(model=current_model)

    output, *args = model(Z_rd=Y, Z_rd_prbs=Z)
    print(f"   Output shape: {output.shape}")

    output = model.predict(Z_rd=Y)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    macs, flops = count_flops(model, {"Z_rd": Y})
    inference_time = measure_inference_time(model, {"Z_rd": Y}, device=device)

    results.append(
        {
            "Model": "RDPDNet (ours)",
            "Order": r"O(L_cnn C^2 k^2 N_chip N_prbs)",
            "Parameters": params,
            "FLOPs": flops,
            "Latency (ms)": inference_time,
        }
    )

    ##############
    # Test DnCNN #
    ##############
    print("\n4. Testing DnCNN...")
    Y = torch.randn(1, c, h, w)

    current_model = dncnn.DnCNN(depth=10, img_channels=2)
    model = DenoiseModel(model=current_model)

    output = model(Z_rd=Y)   # DnCNN: no Z_rd_prbs -> forward returns a tensor
    print(f"   Output shape: {output.shape}")

    output = model.predict(Z_rd=Y)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    macs, flops = count_flops(model, {"Z_rd": Y})
    inference_time = measure_inference_time(model, {"Z_rd": Y}, device=device)

    results.append(
        {
            "Model": "DnCNN",
            "Order": r"O(L' C'^2 k^2 N_chip N_prbs)",
            "Parameters": params,
            "FLOPs": flops,
            "Latency (ms)": inference_time,
        }
    )

    # -----------------------------------------------------------------
    #  Table II — per-frame complexity, params, FLOPs/frame, latency
    #  (manuscript tab_complexity, input 2 x N_chip x N_prbs = 2x512x256).
    #  The MF + CA-CFAR baseline has no learnable parameters; the complete
    #  table (all chains on one device) is scripts/test/complexity_table.py.
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"TABLE II  —  Per-frame complexity  (input {c} x {h} x {w}, batch size 1)")
    print("=" * 92)
    header = f"{'Method':<14}{'Order complexity':<34}{'Params':>10}{'FLOPs/frame':>16}{'Latency':>14}"
    print(header)
    print("-" * 92)

    # Reference row: conventional pipeline (measured in MATLAB, not here).
    print(
        f"{'MF + CA-CFAR':<14}"
        f"{'O(N_chip N_prbs log(N_chip N_prbs))':<34}"
        f"{'0':>10}{'--':>16}{'--':>14}"
    )
    for r in results:
        params_m = r["Parameters"] / 1e6          # millions
        flops_g = r["FLOPs"] / 1e9                 # GFLOPs
        print(
            f"{r['Model']:<14}"
            f"{r['Order']:<34}"
            f"{params_m:>9.3f}M"
            f"{flops_g:>14.3f}G"
            f"{r['Latency (ms)']:>11.2f}ms"
        )

    print("=" * 92)
    print("FLOPs convention: 1 MAC = 2 FLOPs (multiply + add); Conv/Linear layers only.")
    print(f"Device used for latency timing: {device}")
    print(f"Input shape: ({c}, {h}, {w}); batch size: 1; 100 timed runs after 10 warmup.")
    print("Copy Params / FLOPs/frame / Latency into Table II (tab_complexity).")
