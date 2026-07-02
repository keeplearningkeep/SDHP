import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PlantParams:
    dt: float = 0.02
    a: float = 0.2
    b: float = 1.0
    u_max: float = 4.0


@dataclass
class CostWeights:
    qe: float = 10.0
    qv: float = 1.0
    ru: float = 0.02
    qs: float = 0.0
    smax: float = 2.0


def _uniform(
    low: float,
    high: float,
    size,
    generator: torch.Generator,
    dtype=torch.float64,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(size, generator=generator, dtype=dtype)


def quintic_smooth_step_profile(
    t: torch.Tensor,
    t0: float,
    duration: float,
) -> torch.Tensor:
    tau = (t - t0) / duration
    tau = torch.clamp(tau, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def quintic_smooth_step_profile_batch(
    t: torch.Tensor,
    t0: float,
    duration: torch.Tensor,
) -> torch.Tensor:
    duration = duration.to(dtype=t.dtype, device=t.device)
    tau = (t[None, :] - t0) / duration[:, None]
    tau = torch.clamp(tau, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def generate_reference_target(
    n_episodes: int,
    horizon: int,
    dt: float,
    generator: torch.Generator,
    amp_low: float = -0.5,
    amp_high: float = 0.5,
) -> torch.Tensor:
    dtype = torch.float64
    t_grid = torch.arange(horizon, dtype=dtype) * dt
    total_time = horizon * dt

    amp1 = _uniform(amp_low, amp_high, (n_episodes,), generator, dtype=dtype)
    amp2 = _uniform(amp_low, amp_high, (n_episodes,), generator, dtype=dtype)

    t1 = 0.20 * total_time
    t2 = 0.60 * total_time
    a_ref_max = 2.0
    min_duration = 0.9
    max_duration = min(1.7, 0.35 * total_time)

    delta1 = torch.abs(amp1)
    delta2 = torch.abs(amp2 - amp1)
    duration1 = torch.sqrt(5.77 * torch.clamp(delta1, min=1e-6) / a_ref_max)
    duration2 = torch.sqrt(5.77 * torch.clamp(delta2, min=1e-6) / a_ref_max)
    duration1 = torch.clamp(duration1, min=min_duration, max=max_duration)
    duration2 = torch.clamp(duration2, min=min_duration, max=max_duration)

    step1 = quintic_smooth_step_profile_batch(t=t_grid, t0=t1, duration=duration1)
    step2 = quintic_smooth_step_profile_batch(t=t_grid, t0=t2, duration=duration2)
    ref = amp1[:, None] * step1 + (amp2 - amp1)[:, None] * step2

    sine_amp = _uniform(0.0, 0.05, (n_episodes,), generator, dtype=dtype)
    freq = _uniform(0.15, 0.35, (n_episodes,), generator, dtype=dtype)
    phase = _uniform(0.0, 2.0 * math.pi, (n_episodes,), generator, dtype=dtype)
    sine = sine_amp[:, None] * torch.sin(
        2.0 * math.pi * freq[:, None] * t_grid[None, :] + phase[:, None]
    )

    envelope_duration = min(0.8, 0.15 * total_time)
    ramp_in = quintic_smooth_step_profile(t=t_grid, t0=0.0, duration=envelope_duration)
    ramp_out = 1.0 - quintic_smooth_step_profile(
        t=t_grid,
        t0=total_time - envelope_duration,
        duration=envelope_duration,
    )
    return ref + (ramp_in * ramp_out)[None, :] * sine


def generate_disturbance_target(
    n_episodes: int,
    horizon: int,
    dt: float,
    generator: torch.Generator,
) -> torch.Tensor:
    dtype = torch.float64
    t_grid = torch.arange(horizon, dtype=dtype) * dt

    noise = 0.04 * torch.randn(n_episodes, horizon, generator=generator, dtype=dtype)
    amp = _uniform(0.0, 0.02, (n_episodes,), generator, dtype=dtype)
    freq = _uniform(0.2, 1.0, (n_episodes,), generator, dtype=dtype)
    phase = _uniform(0.0, 2.0 * math.pi, (n_episodes,), generator, dtype=dtype)
    sinusoidal = amp[:, None] * torch.sin(
        2.0 * math.pi * freq[:, None] * t_grid[None, :] + phase[:, None]
    )
    return noise + sinusoidal


def generate_initial_states_around_reference(
    ref: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    n_episodes = ref.shape[0]
    dtype = ref.dtype
    r0 = ref[:, 0]
    s1 = r0 + _uniform(-0.15, 0.15, (n_episodes,), generator, dtype=dtype)
    s2 = _uniform(-0.2, 0.2, (n_episodes,), generator, dtype=dtype)
    return torch.stack([s1, s2], dim=-1)


def generate_target_scenarios(
    n_episodes: int,
    horizon: int,
    params: PlantParams,
    seed: int,
    amp_low: float = -0.5,
    amp_high: float = 0.5,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    ref = generate_reference_target(
        n_episodes,
        horizon,
        params.dt,
        generator,
        amp_low=amp_low,
        amp_high=amp_high,
    )
    s0 = generate_initial_states_around_reference(ref=ref, generator=generator)
    dist = generate_disturbance_target(n_episodes, horizon, params.dt, generator)
    return {"s0": s0, "ref": ref, "dist": dist}


def plant_step(
    state: torch.Tensor,
    control: torch.Tensor,
    disturbance: torch.Tensor,
    params: PlantParams,
) -> torch.Tensor:
    s1 = state[:, 0]
    s2 = state[:, 1]
    ds1 = s2
    ds2 = -params.a * s2 - torch.sin(s1) + params.b * control + disturbance
    return torch.stack([s1 + params.dt * ds1, s2 + params.dt * ds2], dim=-1)


def pid_gains(theta: torch.Tensor) -> torch.Tensor:
    return F.softplus(theta)


def pid_control(
    error: torch.Tensor,
    error_integral: torch.Tensor,
    error_derivative: torch.Tensor,
    theta: torch.Tensor,
    params: PlantParams,
) -> torch.Tensor:
    Kp, Ki, Kd = pid_gains(theta)
    raw_u = Kp * error + Ki * error_integral + Kd * error_derivative
    return params.u_max * torch.tanh(raw_u / params.u_max)


def rollout_pid(
    theta: torch.Tensor,
    scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
    return_trajectory: bool = False,
):
    s = scenarios["s0"]
    ref = scenarios["ref"]
    dist = scenarios["dist"]
    n_episodes, horizon = ref.shape

    error_integral = torch.zeros(n_episodes, dtype=s.dtype, device=s.device)
    prev_output = s[:, 0].clone()
    total_cost = torch.zeros((), dtype=s.dtype, device=s.device)
    states, controls, errors = [], [], []

    for t in range(horizon):
        r_t = ref[:, t]
        d_t = dist[:, t]
        error = r_t - s[:, 0]
        if t == 0:
            output_derivative = torch.zeros_like(error)
        else:
            output_derivative = (s[:, 0] - prev_output) / params.dt

        error_integral = error_integral + params.dt * error
        u = pid_control(error, error_integral, -output_derivative, theta, params)
        safety_violation = F.softplus(torch.abs(s[:, 0]) - weights.smax) ** 2
        step_cost = (
            weights.qe * error.pow(2)
            + weights.qv * s[:, 1].pow(2)
            + weights.ru * u.pow(2)
            + weights.qs * safety_violation
        )
        total_cost = total_cost + step_cost.mean()

        if return_trajectory:
            states.append(s)
            controls.append(u)
            errors.append(error)

        prev_output = s[:, 0].clone()
        s = plant_step(s, u, d_t, params)

    average_cost = total_cost / horizon
    if not return_trajectory:
        return average_cost

    return average_cost, {
        "states": torch.stack(states, dim=1),
        "controls": torch.stack(controls, dim=1),
        "errors": torch.stack(errors, dim=1),
        "gains": pid_gains(theta),
    }


def lower_objective(
    log_lambda: torch.Tensor,
    theta: torch.Tensor,
    train_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    lambda_reg = torch.exp(log_lambda)
    gains = pid_gains(theta)
    training_cost = rollout_pid(theta, train_scenarios, params, weights)
    return training_cost + (lambda_reg * gains.pow(2)).sum()


def upper_objective(
    theta: torch.Tensor,
    val_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    return rollout_pid(theta, val_scenarios, params, weights)


def _first_param(params, name: str) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        return params
    params = list(params)
    if len(params) != 1:
        raise ValueError(f"Expected {name} to contain exactly one tensor, got {len(params)}.")
    return params[0]


def lower_objective_fmodel(
    fmodel: torch.nn.Module,
    y_params,
    hparams,
    train_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    theta = fmodel(params=y_params)
    log_lambda = _first_param(hparams, "hparams")
    return lower_objective(log_lambda, theta, train_scenarios, params, weights)


def upper_objective_fmodel(
    fmodel: torch.nn.Module,
    y_params,
    hparams,
    val_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    _ = hparams
    theta = fmodel(params=y_params)
    return upper_objective(theta, val_scenarios, params, weights)


def low_loss_FO(
    fmodel: torch.nn.Module,
    theta,
    y_params,
    hparams,
    ck: float,
    gamma: float,
    train_scenarios: Dict[str, torch.Tensor],
    val_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    reg = 0
    for theta_param, y_param in zip(theta, y_params):
        reg = reg + torch.norm(theta_param - y_param) ** 2

    return upper_objective_fmodel(
        fmodel,
        y_params,
        hparams,
        val_scenarios,
        params,
        weights,
    ) + ck * (
        lower_objective_fmodel(
            fmodel,
            y_params,
            hparams,
            train_scenarios,
            params,
            weights,
        )
        - 0.5 * gamma * reg
    )


def upper_loss_FO(
    fmodel: torch.nn.Module,
    theta,
    y_params,
    hparams,
    ck: float,
    train_scenarios: Dict[str, torch.Tensor],
    val_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    return upper_objective_fmodel(
        fmodel,
        y_params,
        hparams,
        val_scenarios,
        params,
        weights,
    ) + ck * (
        lower_objective_fmodel(
            fmodel,
            y_params,
            hparams,
            train_scenarios,
            params,
            weights,
        )
        - lower_objective_fmodel(
            fmodel,
            theta,
            hparams,
            train_scenarios,
            params,
            weights,
        )
    )


# Backward-compatible aliases for the common F0 spelling.
low_loss_F0 = low_loss_FO
upper_loss_F0 = upper_loss_FO

def copy_parameter_from_list(y, z):
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()

    return y

def lower_foc_constraint(
    log_lambda: torch.Tensor,
    theta: torch.Tensor,
    train_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
) -> torch.Tensor:
    g = lower_objective(log_lambda, theta, train_scenarios, params, weights)
    return torch.autograd.grad(g, theta, create_graph=True, retain_graph=True)[0]


def solve_lower_by_adam(
    log_lambda: torch.Tensor,
    theta_init: torch.Tensor,
    train_scenarios: Dict[str, torch.Tensor],
    params: PlantParams,
    weights: CostWeights,
    lr: float = 0.05,
    steps: int = 500,
    verbose: bool = True,
) -> torch.Tensor:
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for k in range(steps):
        optimizer.zero_grad()
        loss = lower_objective(log_lambda.detach(), theta, train_scenarios, params, weights)
        loss.backward()
        optimizer.step()

        if verbose and (k % 100 == 0 or k == steps - 1):
            with torch.no_grad():
                gains = pid_gains(theta)
                print(
                    f"[Lower Adam] iter={k:04d}, "
                    f"g={loss.item():.6f}, "
                    f"Kp={gains[0].item():.4f}, "
                    f"Ki={gains[1].item():.4f}, "
                    f"Kd={gains[2].item():.4f}"
                )

    return theta.detach().requires_grad_(True)
