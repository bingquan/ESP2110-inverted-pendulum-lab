import json
import math
import numpy as np

NOTEBOOK_PATH = "notebook_for_reference/ESP2110 - Lab Lesson 1 (Setting Up).ipynb"


def load_inverted_pendulum():
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    code = None
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and "class InvertedPendulum" in "".join(cell.get("source", "")):
            code = "".join(cell.get("source", ""))
            break
    if code is None:
        raise RuntimeError("InvertedPendulum class not found in notebook")
    scope = {}
    exec(code, scope)
    return scope["InvertedPendulum"]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(pend, steps, controller, noise_std=0.0, force_disturbance=None, force_limit=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    state_hist = []
    for t in range(steps):
        state = pend.state.copy()
        noisy_state = state.copy()
        if noise_std > 0:
            noisy_state += rng.normal(0, noise_std, size=4)
        force = controller(noisy_state, t)
        if force_disturbance is not None:
            force += force_disturbance(t)
        if force_limit is not None:
            force = float(np.clip(force, -force_limit, force_limit))
        state = pend.step(force)
        if not np.all(np.isfinite(state)):
            return {"ok": False, "reason": "non-finite"}
        if np.any(np.abs(state) > 1e6):
            return {"ok": False, "reason": "blowup"}
        state_hist.append(state.copy())
    return {"ok": True, "states": np.array(state_hist)}


def random_force_controller(scale, rng):
    def c(_state, _t):
        return float(rng.uniform(-scale, scale))
    return c


def zero_controller(_state, _t):
    return 0.0


def make_pid_with_position(theta_ref, kp, kd, ki, kx, kx_dot, dt):
    integ = 0.0

    def controller(state, _t):
        nonlocal integ
        x, x_dot, theta, theta_dot = state
        err = wrap_angle(theta - theta_ref)
        integ += err * dt
        return -kp * err - kd * theta_dot - ki * integ - kx * x - kx_dot * x_dot

    return controller


def tune_pid_with_position(InvertedPendulum, dt=0.02, seed=0):
    rng = np.random.default_rng(seed)
    theta_ref = np.pi
    best = None
    # Coarse grid search
    for kp in [20, 40, 60, 80]:
        for kd in [2, 4, 6, 8]:
            for ki in [0.0, 0.5, 1.0]:
                for kx in [0.1, 0.5, 1.0, 2.0]:
                    for kx_dot in [0.1, 0.5, 1.0, 2.0]:
                        pend = InvertedPendulum(dt=dt)
                        pend.reset()
                        pend.state[0] = 0.0
                        pend.state[1] = 0.0
                        pend.state[2] = theta_ref + 0.1
                        pend.state[3] = 0.0
                        controller = make_pid_with_position(theta_ref, kp, kd, ki, kx, kx_dot, dt)
                        sim = simulate(
                            pend,
                            3000,
                            controller,
                            noise_std=0.0,
                            force_limit=25,
                            rng=rng,
                        )
                        if not sim["ok"]:
                            continue
                        states = sim["states"]
                        err = wrap_angle(states[:, 2] - theta_ref)
                        # Score: angle RMS + cart RMS penalty
                        score = float(np.sqrt(np.mean(err**2)) + 0.05 * np.sqrt(np.mean(states[:, 0] ** 2)))
                        if best is None or score < best["score"]:
                            best = {
                                "kp": kp,
                                "kd": kd,
                                "ki": ki,
                                "kx": kx,
                                "kx_dot": kx_dot,
                                "score": score,
                            }
    return best


def main():
    InvertedPendulum = load_inverted_pendulum()
    rng = np.random.default_rng(0)

    best = tune_pid_with_position(InvertedPendulum, dt=0.02, seed=1)
    if best is None:
        raise RuntimeError("PID tuning failed")

    print("Best PID+pos gains:", best)

    def pid_controller_factory(dt):
        return make_pid_with_position(
            np.pi, best["kp"], best["kd"], best["ki"], best["kx"], best["kx_dot"], dt
        )

    def impulse_disturbance(t):
        return 30.0 if 300 <= t < 360 else 0.0

    def bias_disturbance(_t):
        return 2.0

    def sinusoid_disturbance(t):
        return 5.0 * math.sin(0.02 * t)

    # Stress test conditions
    conditions = []

    # Open-loop tests
    conditions.append({
        "name": "open_loop_long_horizon",
        "steps": 50000,
        "controller": zero_controller,
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": None,
    })
    conditions.append({
        "name": "open_loop_large_forces",
        "steps": 10000,
        "controller": random_force_controller(100, rng),
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": None,
    })
    conditions.append({
        "name": "open_loop_varied_dt_small",
        "steps": 15000,
        "controller": random_force_controller(20, rng),
        "dt": 0.005,
        "noise_std": 0.0,
        "force_disturbance": None,
    })
    conditions.append({
        "name": "open_loop_varied_dt_large",
        "steps": 5000,
        "controller": random_force_controller(20, rng),
        "dt": 0.05,
        "noise_std": 0.0,
        "force_disturbance": None,
    })

    # PID+position tests
    conditions.append({
        "name": "pidpos_nominal",
        "steps": 8000,
        "controller_factory": pid_controller_factory,
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": None,
        "force_limit": 25,
    })
    conditions.append({
        "name": "pidpos_impulse_disturbance",
        "steps": 8000,
        "controller_factory": pid_controller_factory,
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": impulse_disturbance,
        "force_limit": 25,
    })
    conditions.append({
        "name": "pidpos_bias_disturbance",
        "steps": 8000,
        "controller_factory": pid_controller_factory,
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": bias_disturbance,
        "force_limit": 25,
    })
    conditions.append({
        "name": "pidpos_sinusoid_disturbance",
        "steps": 8000,
        "controller_factory": pid_controller_factory,
        "dt": 0.02,
        "noise_std": 0.0,
        "force_disturbance": sinusoid_disturbance,
        "force_limit": 25,
    })
    conditions.append({
        "name": "pidpos_noise_on_state",
        "steps": 8000,
        "controller_factory": pid_controller_factory,
        "dt": 0.02,
        "noise_std": 0.02,
        "force_disturbance": None,
        "force_limit": 25,
    })

    # Run conditions across multiple seeds
    summary = []
    seeds = [0, 1, 2]
    for cond in conditions:
        for seed in seeds:
            pend = InvertedPendulum(dt=cond["dt"])
            pend.reset()
            pend.state[0] = 0.0
            pend.state[1] = 0.0
            pend.state[2] = np.pi + 0.1
            pend.state[3] = 0.0
            if "controller" in cond:
                controller = cond["controller"]
            else:
                controller = cond["controller_factory"](cond["dt"])
            sim = simulate(
                pend,
                cond["steps"],
                controller,
                noise_std=cond.get("noise_std", 0.0),
                force_disturbance=cond.get("force_disturbance"),
                force_limit=cond.get("force_limit"),
                rng=np.random.default_rng(seed),
            )
            label = f"{cond['name']}|seed={seed}"
            if not sim["ok"]:
                summary.append((label, "FAIL", sim["reason"]))
                continue
            states = sim["states"]
            max_abs = np.max(np.abs(states), axis=0)
            err = wrap_angle(states[:, 2] - np.pi)
            err_rms = float(np.sqrt(np.mean(err**2)))
            x_rms = float(np.sqrt(np.mean(states[:, 0] ** 2)))
            summary.append((label, "OK", {"max_abs": max_abs, "theta_rms": err_rms, "x_rms": x_rms}))

    print("\nSummary")
    for name, status, info in summary:
        print(name, status, info)


if __name__ == "__main__":
    main()
