
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(7)
device = torch.device("cpu")

def clean_signal(s: torch.Tensor) -> torch.Tensor:
    return (
        0.80 * torch.sin(2.0 * math.pi * 2.0 * s + 0.20)
        + 0.35 * torch.sin(2.0 * math.pi * 6.0 * s + 0.80)
        + 0.25 * torch.cos(2.0 * math.pi * 11.0 * s)
        + 0.50 * (s - 0.5)
        + 0.25 * torch.exp(-((s - 0.72) / 0.06) ** 2)
    )

def add_asymmetric_outliers(y_clean, sigma=0.04, p_pos=0.08, p_neg=0.02, amp_pos=1.20, amp_neg=0.80):
    gaussian_noise = sigma * torch.randn_like(y_clean)
    u = torch.rand_like(y_clean)
    pos_mask = u < p_pos
    neg_mask = (u >= p_pos) & (u < p_pos + p_neg)

    outlier = torch.zeros_like(y_clean)
    outlier[pos_mask] = amp_pos * (0.5 + torch.rand_like(y_clean[pos_mask]))
    outlier[neg_mask] = -amp_neg * (0.5 + torch.rand_like(y_clean[neg_mask]))

    return y_clean + gaussian_noise + outlier, {
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "outlier": outlier,
        "gaussian_noise": gaussian_noise,
    }

class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=8):
        super().__init__()
        freqs = torch.arange(1, num_frequencies + 1).float().view(1, -1)
        self.register_buffer("freqs", freqs)

    def forward(self, s):
        angles = 2.0 * math.pi * s @ self.freqs
        return torch.cat([s, torch.sin(angles), torch.cos(angles)], dim=-1)

class MLPTimeSeries(nn.Module):
    def __init__(self, num_frequencies=8, hidden_dim=32):
        super().__init__()
        self.features = FourierFeatures(num_frequencies)
        in_dim = 1 + 2 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s):
        return self.net(self.features(s))

def quantile_huber_loss(residual, eta_minus, eta_plus, reduction="mean"):
    eta_minus = torch.as_tensor(eta_minus, device=residual.device, dtype=residual.dtype)
    eta_plus = torch.as_tensor(eta_plus, device=residual.device, dtype=residual.dtype)

    left = -eta_minus * residual - 0.5 * eta_minus**2
    middle = 0.5 * residual**2
    right = eta_plus * residual - 0.5 * eta_plus**2

    loss = torch.where(residual < -eta_minus, left, torch.where(residual > eta_plus, right, middle))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def lower_objective(model, s_train, y_train_corrupt, eta_minus, eta_plus):
    y_hat = model(s_train)
    residual = y_train_corrupt - y_hat
    return quantile_huber_loss(residual, eta_minus, eta_plus, reduction="mean")

def upper_objective_mse(model, s_val, y_val_clean):
    y_hat = model(s_val)
    return torch.mean((y_val_clean - y_hat) ** 2)

def nmse(y_true, y_pred, eps=1e-12):
    return torch.sum((y_true - y_pred) ** 2) / (torch.sum((y_true - y_true.mean()) ** 2) + eps)

def train_lower_with_adam(model, s_train, y_train_corrupt, s_val, y_val_clean,
                          eta_minus=0.25, eta_plus=0.25, lr=1e-3,
                          num_steps=900, log_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"step": [], "lower_qh_train": [], "upper_mse_val": [], "val_nmse": []}

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = lower_objective(model, s_train, y_train_corrupt, eta_minus, eta_plus)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == num_steps:
            model.eval()
            with torch.no_grad():
                val_pred = model(s_val)
                val_mse = torch.mean((y_val_clean - val_pred) ** 2)
                val_nmse = nmse(y_val_clean, val_pred)
            history["step"].append(step)
            history["lower_qh_train"].append(float(loss.detach()))
            history["upper_mse_val"].append(float(val_mse.detach()))
            history["val_nmse"].append(float(val_nmse.detach()))
    return history

def main():
    T_total = 1000
    n_train = 600
    n_val = 200

    s_all = torch.linspace(0.0, 1.0, T_total).view(-1, 1)
    y_clean_all = clean_signal(s_all)

    s_train = s_all[:n_train].to(device)
    s_val = s_all[n_train:n_train+n_val].to(device)
    s_test = s_all[n_train+n_val:].to(device)

    y_train_clean = y_clean_all[:n_train].to(device)
    y_val_clean = y_clean_all[n_train:n_train+n_val].to(device)
    y_test_clean = y_clean_all[n_train+n_val:].to(device)

    y_train_corrupt, train_noise_info = add_asymmetric_outliers(y_train_clean.cpu())
    y_train_corrupt = y_train_corrupt.to(device)

    model = MLPTimeSeries(num_frequencies=8, hidden_dim=32).to(device)

    eta_minus = 0.25
    eta_plus = 0.25

    history = train_lower_with_adam(
        model, s_train, y_train_corrupt, s_val, y_val_clean,
        eta_minus=eta_minus, eta_plus=eta_plus, lr=1e-3, num_steps=900, log_every=100
    )

    model.eval()
    with torch.no_grad():
        y_train_pred = model(s_train)
        y_val_pred = model(s_val)
        y_test_pred = model(s_test)

        final_lower = lower_objective(model, s_train, y_train_corrupt, eta_minus, eta_plus)
        final_val_mse = upper_objective_mse(model, s_val, y_val_clean)
        final_val_nmse = nmse(y_val_clean, y_val_pred)
        final_test_mse = torch.mean((y_test_clean - y_test_pred) ** 2)
        final_test_nmse = nmse(y_test_clean, y_test_pred)

    print(f"Fixed eta_minus = {eta_minus:.3f}, eta_plus = {eta_plus:.3f}")
    print(f"Final lower QH training loss: {final_lower.item():.6f}")
    print(f"Final upper validation MSE:    {final_val_mse.item():.6f}")
    print(f"Final validation NMSE:         {final_val_nmse.item():.6f}")
    print(f"Final clean test MSE:          {final_test_mse.item():.6f}")
    print(f"Final clean test NMSE:         {final_test_nmse.item():.6f}")

    os.makedirs("/mnt/data", exist_ok=True)

    train_s_np = s_train.squeeze().numpy()
    val_s_np = s_val.squeeze().numpy()

    y_train_clean_np = y_train_clean.squeeze().numpy()
    y_train_corrupt_np = y_train_corrupt.squeeze().detach().numpy()
    y_val_clean_np = y_val_clean.squeeze().numpy()
    y_train_pred_np = y_train_pred.squeeze().detach().numpy()
    y_val_pred_np = y_val_pred.squeeze().detach().numpy()

    pos_mask_np = train_noise_info["pos_mask"].squeeze().numpy().astype(bool)
    neg_mask_np = train_noise_info["neg_mask"].squeeze().numpy().astype(bool)

    plt.figure(figsize=(12, 4.8))
    plt.plot(train_s_np, y_train_clean_np, linewidth=2, label="Clean train signal")
    plt.scatter(train_s_np, y_train_corrupt_np, s=10, alpha=0.45, label="Corrupted train observations")
    plt.scatter(train_s_np[pos_mask_np], y_train_corrupt_np[pos_mask_np], s=24, marker="x", label="Positive outliers")
    plt.scatter(train_s_np[neg_mask_np], y_train_corrupt_np[neg_mask_np], s=24, marker="x", label="Negative outliers")
    plt.plot(train_s_np, y_train_pred_np, linewidth=2, label="MLP fit after QH training")
    plt.title("Training signal")
    plt.xlabel("Normalized time")
    plt.ylabel("Signal")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("/mnt/data/train_signal_qh_mlp.png", dpi=180)
    plt.show()

    plt.figure(figsize=(12, 4.8))
    plt.plot(val_s_np, y_val_clean_np, linewidth=2, label="Clean validation signal")
    plt.plot(val_s_np, y_val_pred_np, linewidth=2, label="MLP prediction")
    plt.title("Validation signal")
    plt.xlabel("Normalized time")
    plt.ylabel("Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/val_signal_qh_mlp.png", dpi=180)
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(history["step"], history["lower_qh_train"], marker="o", label="Lower QH train loss")
    plt.plot(history["step"], history["upper_mse_val"], marker="o", label="Upper validation MSE")
    plt.title("Training history")
    plt.xlabel("Adam step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/training_history_qh_mlp.png", dpi=180)
    plt.show()

if __name__ == "__main__":
    main()
