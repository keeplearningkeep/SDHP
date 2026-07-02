import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.io import savemat


from pid_utils import (
    CostWeights,
    PlantParams,
    generate_target_scenarios,
    lower_objective,
    pid_gains,
    rollout_pid,
    upper_objective,
)

torch.set_default_dtype(torch.float64)

# ========================================
class ModelTensorF(torch.nn.Module):
    def __init__(self, tensor):
        super(ModelTensorF, self).__init__()
        self.T = torch.nn.Parameter(tensor)

    def forward(self, i=-1):
        if i == -1:
            return self.T
        return self.T[i]


def update_tensor(hparams, detas, step):
    for p, d in zip(hparams, detas):
        if d is not None:
            with torch.no_grad():
                p.add_(step * d)


def move_scenarios_to_device(scenarios, device):
    return {name: tensor.to(device) for name, tensor in scenarios.items()}


def normlize(P, delta=1):
    device = P[0].device
    dtype = P[0].dtype

    sqnorm = torch.zeros((), device=device, dtype=dtype)
    for p in P:
        sqnorm = sqnorm + p.pow(2).sum()

    scale = 1.0 / torch.sqrt(delta * sqnorm + 1.0)
    P_normed = [p * scale for p in P]  # v / sqrt(δ||v||^2 + 1)
    return P_normed

def append_training_metrics(
    x,
    y,
    train_scenarios,
    val_scenarios,
    test_scenarios,
    params,
    weights,
    x_hist,
    upper_hist,
    lower_hist,
    test_err_hist,
    runtime_hist,
    elapsed_time,
):
    with torch.no_grad():
        lower_loss = lower_objective(
            log_lambda=x(),
            theta=y(),
            train_scenarios=train_scenarios,
            params=params,
            weights=weights,
        )
        upper_loss = upper_objective(
            theta=y(),
            val_scenarios=val_scenarios,
            params=params,
            weights=weights,
        )
        x_hist.append(torch.exp(x()).detach().cpu().numpy())
        upper_hist.append(float(upper_loss.item()))
        lower_hist.append(float(lower_loss.item()))
        test_cost, test_traj = rollout_pid(
            theta=y(),
            scenarios=test_scenarios,
            params=params,
            weights=weights,
            return_trajectory=True,
        )
        test_err_hist.append(float(test_cost.item()))
    if runtime_hist is not None:
        runtime_hist.append(float(elapsed_time))
    return lower_loss, upper_loss, test_cost, test_traj


def record_training_progress(
    k,
    x,
    y,
    train_scenarios,
    val_scenarios,
    test_scenarios,
    params,
    weights,
    x_hist,
    upper_hist,
    lower_hist,
    test_err_hist,
    test_traj,
    runtime_hist=None,
    timer_state=None,
    log_interval=1,
):
    if (k + 1) % log_interval != 0:
        return test_traj

    elapsed_time = time.perf_counter() - timer_state[0] if timer_state is not None else 0.0
    lower_loss, upper_loss, test_cost, test_traj = append_training_metrics(
        x,
        y,
        train_scenarios,
        val_scenarios,
        test_scenarios,
        params,
        weights,
        x_hist,
        upper_hist,
        lower_hist,
        test_err_hist,
        runtime_hist,
        elapsed_time,
    )
    print(
        f"Iter: {k + 1}, "
        f"lambda: {torch.exp(x()).detach().cpu().numpy()} | "
        f"PID gains: {pid_gains(y()).detach().cpu().numpy()} | "
        f"Lower_loss = {lower_loss.item():.6f} | "
        f"Upper_loss f = {upper_loss.item():.6f} | "
        f"Test cost = {test_cost.item():.6f}"
        + (f" | Runtime = {elapsed_time:.4f}s" if elapsed_time is not None else "")
    )
    return test_traj


def _to_numpy_for_mat(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_for_mat(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return np.asarray([_to_numpy_for_mat(item) for item in value])
    return value


def save_algorithm_results(
    algorithm_name,
    x_hist,
    lower_hist,
    upper_hist,
    test_err_hist,
    runtime_hist,
    test_traj,
):
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{algorithm_name}.mat"
    data = {
        "algorithm_name": algorithm_name,
        "x_hist": np.asarray(x_hist),
        "lower_hist": np.asarray(lower_hist, dtype=np.float64),
        "upper_hist": np.asarray(upper_hist, dtype=np.float64),
        "test_err_hist": np.asarray(test_err_hist, dtype=np.float64),
        "runtime_hist": np.asarray(runtime_hist, dtype=np.float64),
        "test_traj": _to_numpy_for_mat(test_traj),
    }
    savemat(output_path, data)
    print(f"Saved training histories and final test trajectory to: {output_path}")


# ============================================================
# 7. Example usage
# ============================================================

def build_experiment():
    params = PlantParams(
        dt=0.02,
        a=0.2,
        b=1.0,
        u_max=4.0,
    )

    horizon = 250

    # Train/validation/test are independently sampled
    # from the same target scenario distribution.
    train_scenarios = generate_target_scenarios(
        n_episodes=16,
        horizon=horizon,
        params=params,
        seed=1,
        amp_low=-0.4,
        amp_high=0.4,
    )

    val_scenarios = generate_target_scenarios(
        n_episodes=16,
        horizon=horizon,
        params=params,
        seed=2,
        amp_low=-0.8,
        amp_high=0.8,
    )

    test_scenarios = generate_target_scenarios(
        n_episodes=128,
        horizon=horizon,
        params=params,
        seed=3,
        amp_low=-0.8,
        amp_high=0.8,
    )

    # Use the same control-performance weights across lower/upper/test.
    common_weights = CostWeights(
        qe=50.0,
        qv=0.1,
        ru=0.001,
        qs=1.0,
        smax=2.0,
    )

    return (
        params,
        train_scenarios,
        val_scenarios,
        test_scenarios,
        common_weights,
    )



if __name__ == "__main__":
    (
        params,
        train_scenarios,
        val_scenarios,
        test_scenarios,
        weights,
    ) = build_experiment()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_scenarios = move_scenarios_to_device(train_scenarios, device)
    val_scenarios = move_scenarios_to_device(val_scenarios, device)
    test_scenarios = move_scenarios_to_device(test_scenarios, device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Upper-level variable:
    #   x = log_lambda
    # lambda = exp(x)
    # ------------------------------------------------------------
    x = ModelTensorF(torch.full((3,), math.log(0.01,math.e), device=device, requires_grad=True))
    # x = ModelTensorF(torch.full((3,), 0.0, device=device, requires_grad=True))

    # ------------------------------------------------------------
    # Lower-level variable:
    #   theta = unconstrained PID parameters
    # gains = softplus(theta)
    # ------------------------------------------------------------
    y = ModelTensorF(torch.zeros(3, device=device, requires_grad=True))
    x_params = list(x.parameters())
    y_params = list(y.parameters())

    print("\nInitial PID gains:", pid_gains(y()).detach().cpu().numpy())
    print("Initial lambda:", torch.exp(x()).detach().cpu().numpy())

    g0 = lower_objective(
        log_lambda=x(),
        theta=y(),
        train_scenarios=train_scenarios,
        params=params,
        weights=weights,
    )

    f0 = upper_objective(
        theta=y(),
        val_scenarios=val_scenarios,
        params=params,
        weights=weights,
    )

    print(f"Initial lower objective g = {g0.item():.6f}")
    print(f"Initial upper objective f = {f0.item():.6f}")

    mode = "SDHP"
    max_iter = 500
    log_interval = 1
    x_hist, upper_hist, lower_hist, test_err_hist = [], [], [], []
    runtime_hist = []
    timer_state = [time.perf_counter()]
    test_traj = None

    _, _, initial_test_cost, test_traj = append_training_metrics(
        x,
        y,
        train_scenarios,
        val_scenarios,
        test_scenarios,
        params,
        weights,
        x_hist,
        upper_hist,
        lower_hist,
        test_err_hist,
        runtime_hist,
        elapsed_time=0.0,
    )
    print(f"Initial test cost = {initial_test_cost.item():.6f}, Runtime = 0.0000s")

    x_lr, y_lr = 0.8, 0.8
    rho = 1.0
    gamma = 1.0
    lamb = [torch.zeros_like(p) for p in y_params]
    Py = [torch.zeros_like(p) for p in y_params]
    Px = [torch.zeros_like(p) for p in x_params]
    exp_x = math.exp(-gamma * x_lr)
    exp_y = math.exp(-gamma * y_lr)

    timer_state[0] = time.perf_counter()
    for k in range(max_iter):
        lower_loss = lower_objective(log_lambda=x(), theta=y(), train_scenarios=train_scenarios, params=params, weights=weights)
        upper_loss = upper_objective(theta=y(), val_scenarios=val_scenarios, params=params, weights=weights)
        dfdy = torch.autograd.grad(upper_loss, y_params, create_graph=True, retain_graph=True, allow_unused=True)
        dgdy = torch.autograd.grad(lower_loss, y_params, create_graph=True, retain_graph=True, allow_unused=True)
        vec = [lam + rho * g for lam, g in zip(lamb, dgdy)]
        dhdy = torch.autograd.grad(dgdy, y_params, grad_outputs=vec, retain_graph=True, allow_unused=True)
        grads_x = torch.autograd.grad(dgdy, x_params, grad_outputs=vec, allow_unused=True)

        grads_y = [g1 + g2 for g1, g2 in zip(dfdy, dhdy)]
        Py = [exp_y * p - y_lr * g for p, g in zip(Py, grads_y)]
        update_tensor(y_params, normlize(Py), y_lr)
        Px = [exp_x * p - x_lr * g for p, g in zip(Px, grads_x)]
        update_tensor(x_params, normlize(Px), x_lr)

        lower_loss = lower_objective(log_lambda=x(), theta=y(), train_scenarios=train_scenarios, params=params, weights=weights)
        dgdy = torch.autograd.grad(lower_loss, y_params, allow_unused=True)
        lamb = [0.4 * lam + rho * g for lam, g in zip(lamb, dgdy)]

        test_traj = record_training_progress(
            k, x, y, train_scenarios, val_scenarios, test_scenarios, params, weights,
            x_hist, upper_hist, lower_hist, test_err_hist, test_traj, runtime_hist,
            timer_state, log_interval=log_interval,
        )

    save_algorithm_results(
        mode,
        x_hist,
        lower_hist,
        upper_hist,
        test_err_hist,
        runtime_hist,
        test_traj,
    )

    # ------------------------------------------------------------
    # Convergence curves after bilevel training
    # ------------------------------------------------------------
    if upper_hist:
        iters = [0] + [log_interval * i for i in range(1, len(upper_hist))]
        plt.figure(figsize=(10, 5))
        plt.plot(iters, upper_hist, label="Upper Loss", linewidth=1.8)
        plt.plot(iters, lower_hist, label="Lower Loss", linewidth=1.8)
        plt.plot(iters, test_err_hist, label="Test Cost", linewidth=1.8)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Bilevel Training Convergence")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()