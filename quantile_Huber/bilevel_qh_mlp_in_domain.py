import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import savemat

torch.set_num_threads(1)

def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(7)
device = torch.device("cpu")

class ModelTensorF(torch.nn.Module):
    def __init__(self, tensor):
        super(ModelTensorF, self).__init__()
        self.T = torch.nn.Parameter(tensor)

    def forward(self, i=-1):
        if i == -1:
            return self.T
        return self.T[i]

def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))

def positive_qh_thresholds(raw_eta: torch.Tensor, eps: float = 1e-6):
    eta = F.softplus(raw_eta) + eps
    return eta[0], eta[1]

def qh_thresholds_from_model(threshold_model: nn.Module):
    return positive_qh_thresholds(threshold_model().view(-1))

def qh_thresholds_from_hparams(hparams):
    hparams = list(hparams)
    if len(hparams) != 1:
        raise ValueError(f"Expected hparams to contain exactly one tensor, got {len(hparams)}.")
    return positive_qh_thresholds(hparams[0].view(-1))

def clean_signal(s: torch.Tensor) -> torch.Tensor:
    """
    Clean latent signal on the full interval [0, 1].
    s shape: [T, 1]
    """
    return (
        0.80 * torch.sin(2.0 * math.pi * 2.0 * s + 0.20)
        + 0.35 * torch.sin(2.0 * math.pi * 6.0 * s + 0.80)
        + 0.25 * torch.cos(2.0 * math.pi * 11.0 * s)
        + 0.50 * (s - 0.5)
        + 0.25 * torch.exp(-((s - 0.72) / 0.06) ** 2)
    )

def add_asymmetric_outliers(
    y_clean: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    sigma: float = 0.1,
    p_pos_background: float = 0.015,
    p_neg_background: float = 0.006,
    pos_hotspot: Tuple[float, float] = (0.58, 0.82),
    neg_hotspot: Tuple[float, float] = (0.15, 0.28),
    p_pos_hotspot: float = 0.1,
    p_neg_hotspot: float = 0.05,
    amp_pos: float = 1.20,
    amp_neg: float = 0.80,
):
    gaussian_noise = sigma * torch.randn_like(y_clean)

    pos_prob = torch.full_like(y_clean, p_pos_background)
    neg_prob = torch.full_like(y_clean, p_neg_background)

    if s is not None:
        s = s.to(device=y_clean.device, dtype=y_clean.dtype)
        pos_hotspot_mask = (s >= pos_hotspot[0]) & (s <= pos_hotspot[1])
        neg_hotspot_mask = (s >= neg_hotspot[0]) & (s <= neg_hotspot[1])
        pos_prob[pos_hotspot_mask] = p_pos_hotspot
        neg_prob[neg_hotspot_mask] = p_neg_hotspot

    pos_mask = torch.rand_like(y_clean) < pos_prob
    neg_mask = (torch.rand_like(y_clean) < neg_prob) & (~pos_mask)

    outlier = torch.zeros_like(y_clean)
    outlier[pos_mask] = amp_pos * (0.5 + torch.rand_like(y_clean[pos_mask]))
    outlier[neg_mask] = -amp_neg * (0.5 + torch.rand_like(y_clean[neg_mask]))

    y_corrupt = y_clean + gaussian_noise + outlier
    info = {
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "outlier": outlier,
        "gaussian_noise": gaussian_noise,
        "pos_prob": pos_prob,
        "neg_prob": neg_prob,
        "pos_hotspot": pos_hotspot,
        "neg_hotspot": neg_hotspot,
    }
    return y_corrupt, info

class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        freqs = torch.arange(1, num_frequencies + 1).float().view(1, -1)
        self.register_buffer("freqs", freqs)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * s @ self.freqs
        return torch.cat([s, torch.sin(angles), torch.cos(angles)], dim=-1)

class MLPTimeSeries(nn.Module):
    def __init__(self, num_frequencies: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.features = FourierFeatures(num_frequencies=num_frequencies)
        in_dim = 1 + 2 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.features(s)
        return self.net(z)

def quantile_huber_loss(
    residual: torch.Tensor,
    eta_minus,
    eta_plus,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Asymmetric quantile Huber loss.

    residual r = y - y_hat.

    rho(r) =
        -eta_minus*r - 0.5*eta_minus^2,    if r < -eta_minus
         0.5*r^2,                          if -eta_minus <= r <= eta_plus
         eta_plus*r - 0.5*eta_plus^2,      if r > eta_plus
    """
    eta_minus = torch.as_tensor(eta_minus, device=residual.device, dtype=residual.dtype)
    eta_plus = torch.as_tensor(eta_plus, device=residual.device, dtype=residual.dtype)

    left = -eta_minus * residual - 0.5 * eta_minus**2
    middle = 0.5 * residual**2
    right = eta_plus * residual - 0.5 * eta_plus**2

    loss = torch.where(
        residual < -eta_minus,
        left,
        torch.where(residual > eta_plus, right, middle),
    )

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def visualize_quantile_huber_shape(
    eta_minus: float = 0.5,
    eta_plus: float = 1.0,
    residual_min: Optional[float] = None,
    residual_max: Optional[float] = None,
    num_points: int = 1000,
    save_path: Optional[str] = "/mnt/data/quantile_huber_shape.png",
    show: bool = True,
):
    """
    Visualize the asymmetric quantile Huber loss as a function of residual r.
    """
    if residual_min is None:
        residual_min = -3.0 * float(eta_minus)
    if residual_max is None:
        residual_max = 3.0 * float(eta_plus)

    residual = torch.linspace(residual_min, residual_max, num_points)
    loss = quantile_huber_loss(residual, eta_minus, eta_plus, reduction="none")

    residual_np = residual.detach().cpu().numpy()
    loss_np = loss.detach().cpu().numpy()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(residual_np, loss_np, linewidth=2, label="Quantile Huber loss")
    plt.axvline(-eta_minus, linestyle="--", linewidth=1.4, color="tab:red", label=r"$-\eta_-$")
    plt.axvline(eta_plus, linestyle="--", linewidth=1.4, color="tab:green", label=r"$\eta_+$")
    plt.axvline(0.0, linestyle=":", linewidth=1.0, color="black", alpha=0.7)
    plt.title(f"Quantile Huber shape: eta_minus={eta_minus:.3f}, eta_plus={eta_plus:.3f}")
    plt.xlabel("Residual r = y - y_hat")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()

def lf(
    x_temp: nn.Module,
    y_temp: nn.Module,
    s_train: torch.Tensor,
    y_train_corrupt: torch.Tensor,
) -> torch.Tensor:
    """
    直接模型版下层目标函数，参考 hyper_cleaning_P2D_FM.py 的 lf(x_temp, y_temp)。
    """
    eta_minus, eta_plus = qh_thresholds_from_model(x_temp)
    y_hat = y_temp(s_train)
    residual = y_train_corrupt - y_hat
    return quantile_huber_loss(residual, eta_minus, eta_plus, reduction="mean")

def uF(
    x_temp: nn.Module,
    y_temp: nn.Module,
    s_val: torch.Tensor,
    y_val_clean: torch.Tensor,
) -> torch.Tensor:
    """
    直接模型版上层目标函数，参考 hyper_cleaning_P2D_FM.py 的 uF(x_temp, y_temp)。
    """
    _ = x_temp
    y_hat = y_temp(s_val)
    return torch.mean((y_val_clean - y_hat) ** 2)

def replace_none_grads(grads, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

def update_tensor(params, deltas, step):
    for p, d in zip(params, deltas):
        if d is not None:
            with torch.no_grad():
                p.add_(step * d)

def normlize(vectors, delta=1.0):
    device_ = vectors[0].device
    dtype_ = vectors[0].dtype
    sqnorm = torch.zeros((), device=device_, dtype=dtype_)
    for v in vectors:
        sqnorm = sqnorm + v.pow(2).sum()
    scale = 1.0 / torch.sqrt(delta * sqnorm + 1.0)
    return [v * scale for v in vectors]

def make_bilevel_history():
    return {
        "step": [],
        "lower_qh_train": [],
        "upper_mse_val": [],
        "eta_minus": [],
        "eta_plus": [],
        "runtime_hist": [],
        "_timer_start": time.perf_counter(),
    }

def append_bilevel_history(
    history,
    step: int,
    threshold_model: nn.Module,
    model: nn.Module,
    s_train: torch.Tensor,
    y_train_corrupt: torch.Tensor,
    s_val: torch.Tensor,
    y_val_clean: torch.Tensor,
):
    with torch.no_grad():
        lower_value = lf(threshold_model, model, s_train, y_train_corrupt)
        upper_value = uF(threshold_model, model, s_val, y_val_clean)
        eta_minus, eta_plus = qh_thresholds_from_model(threshold_model)

    history["step"].append(step)
    history["lower_qh_train"].append(float(lower_value.detach()))
    history["upper_mse_val"].append(float(upper_value.detach()))
    history["eta_minus"].append(float(eta_minus.detach()))
    history["eta_plus"].append(float(eta_plus.detach()))
    elapsed_time = 0.0 if step == 0 else time.perf_counter() - history["_timer_start"]
    history["runtime_hist"].append(float(elapsed_time))

def save_algorithm_history(algorithm_name: str, history, extra_data=None):
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{algorithm_name}.mat"
    data = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in history.items()
        if not key.startswith("_")
    }
    if extra_data is not None:
        data.update(extra_data)
    data["algorithm_name"] = algorithm_name
    savemat(output_path, data)
    print(f"Saved {algorithm_name} history and visualization data to: {output_path}")

def train_bilevel_sdhp(
    model: nn.Module,
    threshold_model: nn.Module,
    s_train: torch.Tensor,
    y_train_corrupt: torch.Tensor,
    s_val: torch.Tensor,
    y_val_clean: torch.Tensor,
    max_iter: int = 500,
    x_lr: float = 0.5,
    y_lr: float = 0.5,
    rho: float = 1.0,
    gamma: float = 1.0,
    log_every: int = 10,
):
    y_params = list(model.parameters())
    x_params = list(threshold_model.parameters())
    lamb = [torch.zeros_like(p) for p in y_params]
    Py = [torch.zeros_like(p) for p in y_params]
    Px = [torch.zeros_like(p) for p in x_params]
    exp_y = math.exp(-gamma * y_lr)
    exp_x = math.exp(-gamma * x_lr)
    history = make_bilevel_history()

    append_bilevel_history(history, 0, threshold_model, model, s_train, y_train_corrupt, s_val, y_val_clean)

    for k in range(max_iter):
        y_params = list(model.parameters())
        x_params = list(threshold_model.parameters())
        lower_value = lf(threshold_model, model, s_train, y_train_corrupt)
        upper_value = uF(threshold_model, model, s_val, y_val_clean)

        dfdy = torch.autograd.grad(
            upper_value,
            y_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        dgdy = torch.autograd.grad(
            lower_value,
            y_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        dfdy = replace_none_grads(dfdy, y_params)
        dgdy = replace_none_grads(dgdy, y_params)

        vec = [lam + rho * g for lam, g in zip(lamb, dgdy)]
        dhdy = torch.autograd.grad(
            dgdy,
            y_params,
            grad_outputs=vec,
            retain_graph=True,
            allow_unused=True,
        )
        grads_x = torch.autograd.grad(
            dgdy,
            x_params,
            grad_outputs=vec,
            allow_unused=True,
        )
        dhdy = replace_none_grads(dhdy, y_params)
        grads_x = replace_none_grads(grads_x, x_params)

        grads_y = [g_f + g_h for g_f, g_h in zip(dfdy, dhdy)]
        Py = [exp_y * p - y_lr * g for p, g in zip(Py, grads_y)]
        update_tensor(y_params, normlize(Py), y_lr)

        Px = [exp_x * p - x_lr * g for p, g in zip(Px, grads_x)]
        update_tensor(x_params, normlize(Px), x_lr)

        lower_new = lf(threshold_model, model, s_train, y_train_corrupt)
        dgdy_new = torch.autograd.grad(lower_new, list(model.parameters()), allow_unused=True)
        dgdy_new = replace_none_grads(dgdy_new, list(model.parameters()))
        lamb = [0.4 * lam + rho * g for lam, g in zip(lamb, dgdy_new)]

        step = k + 1
        if step % log_every == 0 or step == max_iter:
            append_bilevel_history(history, step, threshold_model, model, s_train, y_train_corrupt, s_val, y_val_clean)

    return history

def train_bilevel(
    model: nn.Module,
    threshold_model: nn.Module,
    s_train: torch.Tensor,
    y_train_corrupt: torch.Tensor,
    s_val: torch.Tensor,
    y_val_clean: torch.Tensor,
    max_iter: int = 500,
    log_every: int = 10,
):
    return train_bilevel_sdhp(
        model,
        threshold_model,
        s_train,
        y_train_corrupt,
        s_val,
        y_val_clean,
        max_iter=max_iter,
        log_every=log_every,
    )


def main():
    # ------------------------------------------------------------------
    # In-domain data split:
    # train / validation / test all cover the full interval [0, 1].
    # ------------------------------------------------------------------
    T_total = 1000
    s_all = torch.linspace(0.0, 1.0, T_total).view(-1, 1)
    y_clean_all = clean_signal(s_all)

    idx = torch.arange(T_total)
    block_size = 20
    split_id = idx % block_size
    train_mask = split_id < 12
    val_mask = (split_id >= 12) & (split_id < 16)
    test_mask = split_id >= 16

    pos_hotspot = (0.58, 0.82)
    neg_hotspot = (0.15, 0.28)
    anomaly_interval_mask = (
        ((s_all >= pos_hotspot[0]) & (s_all <= pos_hotspot[1]))
        | ((s_all >= neg_hotspot[0]) & (s_all <= neg_hotspot[1]))
    ).squeeze()

    s_train = s_all[train_mask].to(device)
    s_val = s_all[val_mask].to(device)
    s_test = s_all[test_mask].to(device)

    y_train_clean = y_clean_all[train_mask].to(device)
    y_val_clean = y_clean_all[val_mask].to(device)
    y_test_clean = y_clean_all[test_mask].to(device)

    # Only the training observations are corrupted.
    y_train_corrupt, train_noise_info = add_asymmetric_outliers(
        y_train_clean.cpu(),
        s_train.cpu(),
        pos_hotspot=pos_hotspot,
        neg_hotspot=neg_hotspot,
    )
    y_train_corrupt = y_train_corrupt.to(device)

    model = MLPTimeSeries(num_frequencies=8, hidden_dim=32).to(device)

    initial_eta = torch.tensor([0.5, 0.5], device=device)
    threshold_model = ModelTensorF(inverse_softplus(initial_eta)).to(device)

    mode = "SDHP"
    history = train_bilevel(
        model,
        threshold_model,
        s_train,
        y_train_corrupt,
        s_val,
        y_val_clean,
        max_iter=3000,
        log_every=50,
    )
    model.eval()
    with torch.no_grad():
        eta_minus, eta_plus = qh_thresholds_from_model(threshold_model)

        y_test_pred = model(s_test)
        y_all_pred = model(s_all.to(device)).cpu()

        final_lower = lf(
            x_temp=threshold_model,
            y_temp=model,
            s_train=s_train,
            y_train_corrupt=y_train_corrupt,
        )
        final_val_mse = uF(
            x_temp=threshold_model,
            y_temp=model,
            s_val=s_val,
            y_val_clean=y_val_clean,
        )
        final_test_mse = torch.mean((y_test_clean - y_test_pred) ** 2)

    print(f"In-domain block interpolation split:")
    print(f"Bilevel mode: {mode}")
    print(f"  train points = {int(train_mask.sum())}, val points = {int(val_mask.sum())}, test points = {int(test_mask.sum())}")
    print(f"  validation points inside anomaly intervals = {int((val_mask & anomaly_interval_mask).sum())}")
    print(f"QH thresholds: eta_minus = {eta_minus.item():.3f}, eta_plus = {eta_plus.item():.3f}")
    print(f"Final lower QH training loss: {final_lower.item():.6f}")
    print(f"Final upper validation MSE:    {final_val_mse.item():.6f}")
    print(f"Final clean test MSE:          {final_test_mse.item():.6f}")

    # os.makedirs("quantile_huber_shape.png", exist_ok=True)
    visualize_quantile_huber_shape(
        eta_minus=float(eta_minus.detach().cpu()),
        eta_plus=float(eta_plus.detach().cpu()),
        save_path="quantile_huber_shape.png",
    )

    s_all_np = s_all.squeeze().numpy()
    y_clean_all_np = y_clean_all.squeeze().numpy()
    y_all_pred_np = y_all_pred.squeeze().detach().numpy()

    s_train_np = s_train.squeeze().numpy()
    s_val_np = s_val.squeeze().numpy()
    s_test_np = s_test.squeeze().numpy()

    y_train_corrupt_np = y_train_corrupt.squeeze().detach().numpy()
    y_val_clean_np = y_val_clean.squeeze().numpy()
    y_test_clean_np = y_test_clean.squeeze().numpy()

    pos_mask_np = train_noise_info["pos_mask"].squeeze().numpy().astype(bool)
    neg_mask_np = train_noise_info["neg_mask"].squeeze().numpy().astype(bool)
    outlier_np = train_noise_info["outlier"].squeeze().numpy()
    gaussian_noise_np = train_noise_info["gaussian_noise"].squeeze().numpy()
    pos_prob_np = train_noise_info["pos_prob"].squeeze().numpy()
    neg_prob_np = train_noise_info["neg_prob"].squeeze().numpy()

    visualization_data = {
        "s_all": s_all_np,
        "y_clean_all": y_clean_all_np,
        "y_all_pred": y_all_pred_np,
        "s_train": s_train_np,
        "y_train_clean": y_train_clean.squeeze().numpy(),
        "y_train_corrupt": y_train_corrupt_np,
        "s_val": s_val_np,
        "y_val_clean": y_val_clean_np,
        "s_test": s_test_np,
        "y_test_clean": y_test_clean_np,
        "train_mask": train_mask.numpy().astype(np.uint8),
        "val_mask": val_mask.numpy().astype(np.uint8),
        "test_mask": test_mask.numpy().astype(np.uint8),
        "anomaly_interval_mask": anomaly_interval_mask.numpy().astype(np.uint8),
        "pos_mask_train": pos_mask_np.astype(np.uint8),
        "neg_mask_train": neg_mask_np.astype(np.uint8),
        "s_pos_outlier": s_train_np[pos_mask_np],
        "y_pos_outlier": y_train_corrupt_np[pos_mask_np],
        "s_neg_outlier": s_train_np[neg_mask_np],
        "y_neg_outlier": y_train_corrupt_np[neg_mask_np],
        "train_outlier": outlier_np,
        "train_gaussian_noise": gaussian_noise_np,
        "pos_outlier_prob": pos_prob_np,
        "neg_outlier_prob": neg_prob_np,
        "pos_hotspot": np.asarray(train_noise_info["pos_hotspot"], dtype=np.float64),
        "neg_hotspot": np.asarray(train_noise_info["neg_hotspot"], dtype=np.float64),
        "final_eta_minus": np.asarray(float(eta_minus.detach().cpu()), dtype=np.float64),
        "final_eta_plus": np.asarray(float(eta_plus.detach().cpu()), dtype=np.float64),
        "final_lower_qh_train": np.asarray(float(final_lower.detach().cpu()), dtype=np.float64),
        "final_upper_mse_val": np.asarray(float(final_val_mse.detach().cpu()), dtype=np.float64),
        "final_clean_test_mse": np.asarray(float(final_test_mse.detach().cpu()), dtype=np.float64),
    }
    save_algorithm_history(mode, history, visualization_data)

    plt.figure(figsize=(12, 5.0))
    plt.axvspan(
        train_noise_info["pos_hotspot"][0],
        train_noise_info["pos_hotspot"][1],
        color="tab:red",
        alpha=0.08,
        label="Positive-outlier hotspot",
    )
    plt.axvspan(
        train_noise_info["neg_hotspot"][0],
        train_noise_info["neg_hotspot"][1],
        color="tab:blue",
        alpha=0.08,
        label="Negative-outlier hotspot",
    )
    plt.plot(s_all_np, y_clean_all_np, linewidth=2, label="Clean latent signal")
    plt.plot(s_all_np, y_all_pred_np, linewidth=2, label="MLP fit after QH training")
    plt.scatter(s_train_np, y_train_corrupt_np, s=9, alpha=0.35, label="Corrupted train observations")
    plt.scatter(s_train_np[pos_mask_np], y_train_corrupt_np[pos_mask_np], s=22, marker="x", label="Positive outliers")
    plt.scatter(s_train_np[neg_mask_np], y_train_corrupt_np[neg_mask_np], s=22, marker="x", label="Negative outliers")
    plt.title("In-domain block interpolation: train points cover the full interval")
    plt.xlabel("Normalized time")
    plt.ylabel("Signal")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("/mnt/data/in_domain_train_signal_qh_mlp.png", dpi=180)
    plt.show()

    plt.figure(figsize=(9, 4.8))
    plt.plot(history["step"], history["eta_minus"], marker="o", label="eta_minus")
    plt.plot(history["step"], history["eta_plus"], marker="o", label="eta_plus")
    plt.title(f"{mode} threshold trajectories")
    plt.xlabel("Bilevel iteration")
    plt.ylabel("QH threshold")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/mnt/data/in_domain_{mode.lower()}_thresholds_qh_mlp.png", dpi=180)
    plt.show()

if __name__ == "__main__":
    main()
