#!/usr/bin/env python3
"""
Tester Agent for ESP2110 Inverted Pendulum Lab

An interactive agent that tests and explores the inverted pendulum system
to learn about control systems through hands-on experimentation.

Usage:
    python tester.py                    # Interactive menu
    python tester.py --experiment <n>   # Run specific experiment (1-8)
    python tester.py --all              # Run all experiments
"""

import json
import math
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Path to the notebook containing the InvertedPendulum class
NOTEBOOK_PATH = "updated_notebooks/ESP2110 - Lab Lesson 1 (Setting Up).ipynb"
FALLBACK_PATH = "notebook_for_reference/ESP2110 - Lab Lesson 1 (Setting Up).ipynb"


class TesterAgent:
    """
    An educational agent that interacts with the inverted pendulum system
    to explore and learn control systems concepts.
    """

    def __init__(self, verbose=True, plot=True):
        self.verbose = verbose
        self.plot_enabled = plot and HAS_MATPLOTLIB
        self.InvertedPendulum = self._load_pendulum_class()
        self.experiment_results = {}

    def _load_pendulum_class(self):
        """Load the InvertedPendulum class from the notebook."""
        for path in [NOTEBOOK_PATH, FALLBACK_PATH]:
            try:
                with open(path) as f:
                    nb = json.load(f)
                for cell in nb["cells"]:
                    if cell.get("cell_type") == "code":
                        source = "".join(cell.get("source", ""))
                        if "class InvertedPendulum" in source:
                            scope = {"np": np, "numpy": np}
                            exec(source, scope)
                            if self.verbose:
                                print(f"✓ Loaded InvertedPendulum from {path}")
                            return scope["InvertedPendulum"]
            except FileNotFoundError:
                continue
        raise RuntimeError("Could not find InvertedPendulum class in any notebook")

    def log(self, message):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    @staticmethod
    def wrap_angle(angle):
        """Wrap angle to [-pi, pi] range."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def simulate(self, pend, steps, controller, dt=0.02, noise_std=0.0,
                 disturbance=None, force_limit=None):
        """
        Run a simulation with the given controller.

        Returns:
            dict with 'ok' status and 'states' history
        """
        rng = np.random.default_rng(42)
        states = []
        forces = []

        for t in range(steps):
            state = pend.state.copy()

            # Add measurement noise if specified
            noisy_state = state.copy()
            if noise_std > 0:
                noisy_state += rng.normal(0, noise_std, size=4)

            # Compute control force
            force = controller(noisy_state, t, dt)

            # Add disturbance if specified
            if disturbance is not None:
                force += disturbance(t)

            # Apply force limits if specified
            if force_limit is not None:
                force = float(np.clip(force, -force_limit, force_limit))

            forces.append(force)

            # Step the simulation
            new_state = pend.step(force)

            # Check for simulation failure
            if not np.all(np.isfinite(new_state)):
                return {"ok": False, "reason": "non-finite values",
                        "states": np.array(states), "forces": np.array(forces)}
            if np.any(np.abs(new_state) > 1e6):
                return {"ok": False, "reason": "state blowup",
                        "states": np.array(states), "forces": np.array(forces)}

            states.append(new_state.copy())

        return {"ok": True, "states": np.array(states), "forces": np.array(forces)}

    def analyze_results(self, states, theta_ref=np.pi):
        """Compute performance metrics from simulation states."""
        x, x_dot, theta, theta_dot = states.T

        angle_error = self.wrap_angle(theta - theta_ref)

        return {
            "angle_rms": float(np.sqrt(np.mean(angle_error**2))),
            "angle_max": float(np.max(np.abs(angle_error))),
            "position_rms": float(np.sqrt(np.mean(x**2))),
            "position_max": float(np.max(np.abs(x))),
            "velocity_max": float(np.max(np.abs(x_dot))),
            "angular_velocity_max": float(np.max(np.abs(theta_dot))),
            "final_angle_error": float(np.abs(angle_error[-1])),
            "final_position": float(x[-1]),
            "settled": float(np.abs(angle_error[-1])) < 0.05,
        }

    def plot_results(self, states, forces, title="Simulation Results", dt=0.02):
        """Plot simulation results."""
        if not self.plot_enabled:
            self.log("(Plotting disabled or matplotlib not available)")
            return

        t = np.arange(len(states)) * dt
        x, x_dot, theta, theta_dot = states.T

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)

        # Cart position
        axes[0, 0].plot(t, x, 'b-', linewidth=1)
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Cart Position')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Cart velocity
        axes[0, 1].plot(t, x_dot, 'b-', linewidth=1)
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Cart Velocity')
        axes[0, 1].grid(True, alpha=0.3)

        # Pole angle
        axes[1, 0].plot(t, np.degrees(theta), 'r-', linewidth=1)
        axes[1, 0].axhline(y=180, color='g', linestyle='--', alpha=0.7, label='Upright (180°)')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].set_title('Pole Angle')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Angular velocity
        axes[1, 1].plot(t, np.degrees(theta_dot), 'r-', linewidth=1)
        axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[1, 1].set_title('Pole Angular Velocity')
        axes[1, 1].grid(True, alpha=0.3)

        # Control force
        axes[2, 0].plot(t, forces, 'g-', linewidth=1)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Force (N)')
        axes[2, 0].set_title('Control Force')
        axes[2, 0].grid(True, alpha=0.3)

        # Angle error from upright
        angle_error = self.wrap_angle(theta - np.pi)
        axes[2, 1].plot(t, np.degrees(angle_error), 'm-', linewidth=1)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Error (degrees)')
        axes[2, 1].set_title('Angle Error from Upright')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    # =========================================================================
    # CONTROLLERS
    # =========================================================================

    def no_control(self, state, t, dt):
        """No control - let the system evolve naturally."""
        return 0.0

    def make_p_controller(self, kp, theta_ref=np.pi):
        """Create a proportional controller."""
        def controller(state, t, dt):
            x, x_dot, theta, theta_dot = state
            error = self.wrap_angle(theta - theta_ref)
            return -kp * error
        return controller

    def make_pd_controller(self, kp, kd, theta_ref=np.pi):
        """Create a proportional-derivative controller."""
        def controller(state, t, dt):
            x, x_dot, theta, theta_dot = state
            error = self.wrap_angle(theta - theta_ref)
            return -kp * error - kd * theta_dot
        return controller

    def make_pid_controller(self, kp, kd, ki, theta_ref=np.pi):
        """Create a PID controller."""
        integral = [0.0]  # Use list to allow modification in closure

        def controller(state, t, dt):
            x, x_dot, theta, theta_dot = state
            error = self.wrap_angle(theta - theta_ref)
            integral[0] += error * dt
            # Anti-windup: limit integral term
            integral[0] = np.clip(integral[0], -10, 10)
            return -kp * error - kd * theta_dot - ki * integral[0]
        return controller

    def make_full_state_controller(self, kp, kd, ki, kx, kx_dot, theta_ref=np.pi, x_ref=0.0):
        """Create a full-state feedback controller with position control."""
        integral = [0.0]

        def controller(state, t, dt):
            x, x_dot, theta, theta_dot = state
            error = self.wrap_angle(theta - theta_ref)
            integral[0] += error * dt
            integral[0] = np.clip(integral[0], -10, 10)

            force = (-kp * error
                     - kd * theta_dot
                     - ki * integral[0]
                     - kx * (x - x_ref)
                     - kx_dot * x_dot)
            return force
        return controller

    # =========================================================================
    # EXPERIMENTS
    # =========================================================================

    def experiment_1_open_loop(self):
        """
        Experiment 1: Open-Loop Behavior

        Learn: What happens when we don't control the pendulum?
        Key concept: Unstable equilibrium
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 1: Open-Loop Behavior (No Control)")
        self.log("="*70)
        self.log("\nGoal: Observe the natural behavior of an uncontrolled pendulum")
        self.log("Key Concept: The upright position is an UNSTABLE equilibrium\n")

        pend = self.InvertedPendulum(dt=0.02)
        pend.reset()

        # Start near upright with small perturbation
        pend.state = np.array([0.0, 0.0, np.pi + 0.05, 0.0])

        self.log(f"Initial state: cart at x=0, pole at {np.degrees(pend.state[2]):.1f}° (upright=180°)")
        self.log("Running simulation with NO control force...")

        result = self.simulate(pend, 500, self.no_control)

        if result["ok"]:
            metrics = self.analyze_results(result["states"])
            self.log(f"\nResults after {500 * 0.02:.1f} seconds:")
            self.log(f"  - Final angle: {np.degrees(result['states'][-1, 2]):.1f}°")
            self.log(f"  - Maximum angle deviation: {np.degrees(metrics['angle_max']):.1f}°")
            self.log(f"  - Pole settled upright: {metrics['settled']}")

            self.log("\n📚 LEARNING:")
            self.log("   The pendulum falls because the upright position is UNSTABLE.")
            self.log("   Any small perturbation grows exponentially without control.")
            self.log("   This is why we need feedback control!")

            self.plot_results(result["states"], result["forces"],
                            "Experiment 1: Open-Loop (No Control)")
        else:
            self.log(f"Simulation failed: {result['reason']}")

        self.experiment_results["exp1_open_loop"] = result
        return result

    def experiment_2_p_control(self):
        """
        Experiment 2: Proportional Control

        Learn: The simplest form of feedback control
        Key concept: P-control alone is often insufficient for stabilization
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 2: Proportional (P) Control")
        self.log("="*70)
        self.log("\nGoal: Test if proportional control alone can stabilize the pendulum")
        self.log("Controller: F = -Kp * (theta - theta_ref)")
        self.log("Key Concept: P-control responds to error magnitude\n")

        gains_to_test = [10, 30, 50, 100]
        results = {}

        for kp in gains_to_test:
            pend = self.InvertedPendulum(dt=0.02)
            pend.state = np.array([0.0, 0.0, np.pi + 0.1, 0.0])

            controller = self.make_p_controller(kp)
            result = self.simulate(pend, 2000, controller, force_limit=50)

            if result["ok"]:
                metrics = self.analyze_results(result["states"])
                results[kp] = {"result": result, "metrics": metrics}
                self.log(f"Kp={kp:3d}: Angle RMS={np.degrees(metrics['angle_rms']):6.2f}°, "
                        f"Settled={metrics['settled']}")
            else:
                results[kp] = {"result": result, "metrics": None}
                self.log(f"Kp={kp:3d}: FAILED - {result['reason']}")

        self.log("\n📚 LEARNING:")
        self.log("   P-control applies force proportional to the error.")
        self.log("   Higher Kp = stronger response, but can cause oscillations.")
        self.log("   P-control alone typically cannot stabilize the inverted pendulum")
        self.log("   because it doesn't account for velocity (no damping).")

        # Plot best result
        best_kp = max((kp for kp in gains_to_test if results[kp]["metrics"]),
                     key=lambda k: -results[k]["metrics"]["angle_rms"] if results[k]["metrics"] else float('inf'),
                     default=None)
        if best_kp and results[best_kp]["result"]["ok"]:
            self.plot_results(results[best_kp]["result"]["states"],
                            results[best_kp]["result"]["forces"],
                            f"Experiment 2: P-Control (Kp={best_kp})")

        self.experiment_results["exp2_p_control"] = results
        return results

    def experiment_3_pd_control(self):
        """
        Experiment 3: Proportional-Derivative Control

        Learn: Adding derivative term for damping
        Key concept: D-term predicts future error and adds damping
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 3: Proportional-Derivative (PD) Control")
        self.log("="*70)
        self.log("\nGoal: Add derivative control for better stability")
        self.log("Controller: F = -Kp * error - Kd * angular_velocity")
        self.log("Key Concept: D-term adds damping to reduce oscillations\n")

        # Test different Kd values with fixed Kp
        kp = 50
        kd_values = [1, 5, 10, 20]
        results = {}

        for kd in kd_values:
            pend = self.InvertedPendulum(dt=0.02)
            pend.state = np.array([0.0, 0.0, np.pi + 0.1, 0.0])

            controller = self.make_pd_controller(kp, kd)
            result = self.simulate(pend, 3000, controller, force_limit=50)

            if result["ok"]:
                metrics = self.analyze_results(result["states"])
                results[kd] = {"result": result, "metrics": metrics}
                self.log(f"Kp={kp}, Kd={kd:2d}: Angle RMS={np.degrees(metrics['angle_rms']):6.3f}°, "
                        f"Final error={np.degrees(metrics['final_angle_error']):6.3f}°, "
                        f"Settled={metrics['settled']}")
            else:
                results[kd] = {"result": result, "metrics": None}
                self.log(f"Kp={kp}, Kd={kd:2d}: FAILED - {result['reason']}")

        self.log("\n📚 LEARNING:")
        self.log("   The derivative term (Kd * angular_velocity) adds DAMPING.")
        self.log("   It predicts where the system is heading and counteracts it.")
        self.log("   PD control can stabilize the pendulum but may have steady-state error.")
        self.log("   Too much Kd can slow the response; too little causes oscillation.")

        # Plot best result
        best_kd = min((kd for kd in kd_values if results[kd]["metrics"]),
                     key=lambda k: results[k]["metrics"]["angle_rms"] if results[k]["metrics"] else float('inf'),
                     default=None)
        if best_kd and results[best_kd]["result"]["ok"]:
            self.plot_results(results[best_kd]["result"]["states"],
                            results[best_kd]["result"]["forces"],
                            f"Experiment 3: PD-Control (Kp={kp}, Kd={best_kd})")

        self.experiment_results["exp3_pd_control"] = results
        return results

    def experiment_4_pid_control(self):
        """
        Experiment 4: Full PID Control

        Learn: Adding integral term for steady-state error elimination
        Key concept: I-term accumulates error over time
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 4: PID Control")
        self.log("="*70)
        self.log("\nGoal: Eliminate steady-state error with integral control")
        self.log("Controller: F = -Kp * error - Kd * d(error)/dt - Ki * integral(error)")
        self.log("Key Concept: I-term eliminates persistent errors\n")

        # Test different Ki values with fixed Kp and Kd
        kp, kd = 50, 10
        ki_values = [0, 0.5, 1.0, 2.0, 5.0]
        results = {}

        for ki in ki_values:
            pend = self.InvertedPendulum(dt=0.02)
            pend.state = np.array([0.0, 0.0, np.pi + 0.1, 0.0])

            controller = self.make_pid_controller(kp, kd, ki)
            result = self.simulate(pend, 4000, controller, force_limit=50)

            if result["ok"]:
                metrics = self.analyze_results(result["states"])
                results[ki] = {"result": result, "metrics": metrics}
                self.log(f"Kp={kp}, Kd={kd}, Ki={ki:3.1f}: "
                        f"Angle RMS={np.degrees(metrics['angle_rms']):6.3f}°, "
                        f"Final error={np.degrees(metrics['final_angle_error']):6.4f}°, "
                        f"Settled={metrics['settled']}")
            else:
                results[ki] = {"result": result, "metrics": None}
                self.log(f"Kp={kp}, Kd={kd}, Ki={ki:3.1f}: FAILED - {result['reason']}")

        self.log("\n📚 LEARNING:")
        self.log("   The integral term accumulates error over time.")
        self.log("   It eliminates steady-state error but can cause 'integral windup'.")
        self.log("   Too much Ki causes overshoot and oscillation.")
        self.log("   Tuning PID gains is a balance: Kp (response), Kd (damping), Ki (precision).")

        # Plot result with Ki=1.0
        if 1.0 in results and results[1.0]["result"]["ok"]:
            self.plot_results(results[1.0]["result"]["states"],
                            results[1.0]["result"]["forces"],
                            f"Experiment 4: PID Control (Kp={kp}, Kd={kd}, Ki=1.0)")

        self.experiment_results["exp4_pid_control"] = results
        return results

    def experiment_5_cart_position_control(self):
        """
        Experiment 5: Full State Feedback (Pole + Cart Position)

        Learn: Controlling both the pole angle AND cart position
        Key concept: Coupled control objectives
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 5: Full State Feedback Control")
        self.log("="*70)
        self.log("\nGoal: Control both pole angle AND cart position")
        self.log("Controller: F = -Kp*θ_err - Kd*θ_dot - Ki*∫θ_err - Kx*x - Kx_dot*x_dot")
        self.log("Key Concept: Balancing coupled control objectives\n")

        # Good gains from stress testing
        kp, kd, ki = 60, 6, 0.5
        kx, kx_dot = 1.0, 1.0

        pend = self.InvertedPendulum(dt=0.02)
        pend.state = np.array([0.5, 0.0, np.pi + 0.1, 0.0])  # Start with cart offset

        self.log(f"Initial cart position: {pend.state[0]:.1f}m (we want it at 0)")
        self.log(f"Initial pole angle: {np.degrees(pend.state[2]):.1f}° (we want 180°)")

        controller = self.make_full_state_controller(kp, kd, ki, kx, kx_dot)
        result = self.simulate(pend, 5000, controller, force_limit=30)

        if result["ok"]:
            metrics = self.analyze_results(result["states"])
            self.log(f"\nResults after {5000 * 0.02:.1f} seconds:")
            self.log(f"  - Final cart position: {metrics['final_position']:.4f}m")
            self.log(f"  - Final angle error: {np.degrees(metrics['final_angle_error']):.4f}°")
            self.log(f"  - Position RMS: {metrics['position_rms']:.4f}m")
            self.log(f"  - Angle RMS: {np.degrees(metrics['angle_rms']):.4f}°")

            self.log("\n📚 LEARNING:")
            self.log("   Full state feedback controls ALL state variables.")
            self.log("   Adding cart position terms (Kx, Kx_dot) keeps the cart centered.")
            self.log("   There's a trade-off: aggressive position control can destabilize the pole.")
            self.log("   Good tuning balances pole stability with cart positioning.")

            self.plot_results(result["states"], result["forces"],
                            "Experiment 5: Full State Feedback Control")
        else:
            self.log(f"Simulation failed: {result['reason']}")

        self.experiment_results["exp5_full_state"] = result
        return result

    def experiment_6_disturbance_rejection(self):
        """
        Experiment 6: Disturbance Rejection

        Learn: How controllers handle external disturbances
        Key concept: Robustness to unexpected forces
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 6: Disturbance Rejection")
        self.log("="*70)
        self.log("\nGoal: Test controller robustness to external disturbances")
        self.log("Key Concept: A good controller rejects disturbances and returns to setpoint\n")

        kp, kd, ki, kx, kx_dot = 60, 6, 0.5, 1.0, 1.0

        disturbances = {
            "none": lambda t: 0.0,
            "impulse": lambda t: 20.0 if 100 <= t < 110 else 0.0,
            "constant_bias": lambda t: 3.0 if t >= 100 else 0.0,
            "sinusoidal": lambda t: 5.0 * math.sin(0.1 * t) if t >= 100 else 0.0,
        }

        results = {}
        for name, disturbance in disturbances.items():
            pend = self.InvertedPendulum(dt=0.02)
            pend.state = np.array([0.0, 0.0, np.pi, 0.0])  # Start at equilibrium

            controller = self.make_full_state_controller(kp, kd, ki, kx, kx_dot)
            result = self.simulate(pend, 4000, controller, disturbance=disturbance, force_limit=30)

            if result["ok"]:
                metrics = self.analyze_results(result["states"])
                results[name] = {"result": result, "metrics": metrics}
                self.log(f"{name:15s}: Angle RMS={np.degrees(metrics['angle_rms']):6.3f}°, "
                        f"Position RMS={metrics['position_rms']:.3f}m, "
                        f"Recovered={metrics['settled']}")
            else:
                results[name] = {"result": result, "metrics": None}
                self.log(f"{name:15s}: FAILED - {result['reason']}")

        self.log("\n📚 LEARNING:")
        self.log("   Controllers must reject disturbances to maintain stability.")
        self.log("   Impulse: short, sharp force - tests transient response.")
        self.log("   Constant bias: persistent force - tests integral action.")
        self.log("   Sinusoidal: periodic force - tests frequency response.")
        self.log("   The integral term (Ki) is crucial for rejecting constant biases.")

        # Plot impulse response
        if "impulse" in results and results["impulse"]["result"]["ok"]:
            self.plot_results(results["impulse"]["result"]["states"],
                            results["impulse"]["result"]["forces"],
                            "Experiment 6: Impulse Disturbance Rejection")

        self.experiment_results["exp6_disturbance"] = results
        return results

    def experiment_7_noise_sensitivity(self):
        """
        Experiment 7: Measurement Noise Sensitivity

        Learn: Real sensors have noise - how does this affect control?
        Key concept: Noise amplification, especially in derivative terms
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 7: Measurement Noise Sensitivity")
        self.log("="*70)
        self.log("\nGoal: Understand how sensor noise affects controller performance")
        self.log("Key Concept: Derivative terms amplify high-frequency noise\n")

        kp, kd, ki, kx, kx_dot = 60, 6, 0.5, 1.0, 1.0
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]

        results = {}
        for noise_std in noise_levels:
            pend = self.InvertedPendulum(dt=0.02)
            pend.state = np.array([0.0, 0.0, np.pi + 0.05, 0.0])

            controller = self.make_full_state_controller(kp, kd, ki, kx, kx_dot)
            result = self.simulate(pend, 4000, controller, noise_std=noise_std, force_limit=30)

            if result["ok"]:
                metrics = self.analyze_results(result["states"])
                force_std = np.std(result["forces"])
                results[noise_std] = {"result": result, "metrics": metrics, "force_std": force_std}
                self.log(f"Noise σ={noise_std:.2f}: Angle RMS={np.degrees(metrics['angle_rms']):6.3f}°, "
                        f"Force σ={force_std:.2f}N, Settled={metrics['settled']}")
            else:
                results[noise_std] = {"result": result, "metrics": None}
                self.log(f"Noise σ={noise_std:.2f}: FAILED - {result['reason']}")

        self.log("\n📚 LEARNING:")
        self.log("   Real sensors (encoders, accelerometers) have measurement noise.")
        self.log("   The derivative term (Kd) amplifies high-frequency noise.")
        self.log("   This causes 'jittery' control forces and reduced performance.")
        self.log("   Solutions: filtering, observers (Kalman filter), or reduced Kd.")

        # Plot high noise case
        if 0.05 in results and results[0.05]["result"]["ok"]:
            self.plot_results(results[0.05]["result"]["states"],
                            results[0.05]["result"]["forces"],
                            "Experiment 7: Effect of Measurement Noise (σ=0.05)")

        self.experiment_results["exp7_noise"] = results
        return results

    def experiment_8_gain_tuning_exploration(self):
        """
        Experiment 8: Systematic Gain Tuning

        Learn: How to tune controller gains systematically
        Key concept: Trade-offs in controller design
        """
        self.log("\n" + "="*70)
        self.log("EXPERIMENT 8: Systematic Gain Tuning")
        self.log("="*70)
        self.log("\nGoal: Explore the effect of different gain combinations")
        self.log("Key Concept: There are trade-offs between speed, stability, and effort\n")

        # Grid search over Kp and Kd
        kp_values = [20, 40, 60, 80]
        kd_values = [2, 4, 6, 8, 10]
        ki = 0.5
        kx, kx_dot = 0.5, 0.5

        self.log("Testing gain combinations (Kp x Kd grid):")
        self.log("-" * 60)

        results = {}
        best_score = float('inf')
        best_gains = None

        for kp in kp_values:
            row = []
            for kd in kd_values:
                pend = self.InvertedPendulum(dt=0.02)
                pend.state = np.array([0.2, 0.0, np.pi + 0.1, 0.0])

                controller = self.make_full_state_controller(kp, kd, ki, kx, kx_dot)
                result = self.simulate(pend, 3000, controller, force_limit=30)

                if result["ok"]:
                    metrics = self.analyze_results(result["states"])
                    # Score: balance angle error, position error, and control effort
                    score = (metrics["angle_rms"] +
                            0.1 * metrics["position_rms"] +
                            0.01 * np.mean(np.abs(result["forces"])))
                    results[(kp, kd)] = {"result": result, "metrics": metrics, "score": score}
                    row.append(f"{score:.3f}")

                    if score < best_score:
                        best_score = score
                        best_gains = (kp, kd)
                else:
                    results[(kp, kd)] = {"result": result, "metrics": None}
                    row.append("FAIL ")

            self.log(f"Kp={kp:2d}: " + " | ".join(row))

        self.log("-" * 60)
        if best_gains:
            self.log(f"\nBest gains: Kp={best_gains[0]}, Kd={best_gains[1]} (score={best_score:.4f})")

        self.log("\n📚 LEARNING:")
        self.log("   Controller tuning involves balancing multiple objectives:")
        self.log("   - Response speed (higher Kp = faster, but can be unstable)")
        self.log("   - Damping (higher Kd = less oscillation, but slower)")
        self.log("   - Control effort (lower gains = less actuator wear)")
        self.log("   - Robustness (moderate gains often more robust)")
        self.log("   Systematic approaches: Ziegler-Nichols, LQR, pole placement")

        # Plot best result
        if best_gains and results[best_gains]["result"]["ok"]:
            self.plot_results(results[best_gains]["result"]["states"],
                            results[best_gains]["result"]["forces"],
                            f"Experiment 8: Best Gains (Kp={best_gains[0]}, Kd={best_gains[1]})")

        self.experiment_results["exp8_tuning"] = results
        return results

    def run_all_experiments(self):
        """Run all experiments in sequence."""
        self.log("\n" + "="*70)
        self.log("RUNNING ALL EXPERIMENTS")
        self.log("="*70)

        experiments = [
            self.experiment_1_open_loop,
            self.experiment_2_p_control,
            self.experiment_3_pd_control,
            self.experiment_4_pid_control,
            self.experiment_5_cart_position_control,
            self.experiment_6_disturbance_rejection,
            self.experiment_7_noise_sensitivity,
            self.experiment_8_gain_tuning_exploration,
        ]

        for i, exp in enumerate(experiments, 1):
            try:
                exp()
            except Exception as e:
                self.log(f"\nExperiment {i} failed with error: {e}")

        self.log("\n" + "="*70)
        self.log("ALL EXPERIMENTS COMPLETE")
        self.log("="*70)
        self.print_summary()

    def print_summary(self):
        """Print a summary of key learnings."""
        self.log("\n" + "="*70)
        self.log("SUMMARY: KEY CONTROL SYSTEMS CONCEPTS")
        self.log("="*70)
        self.log("""
1. OPEN-LOOP vs CLOSED-LOOP
   - Open-loop: No feedback, unstable systems fail
   - Closed-loop: Feedback enables stabilization

2. P-CONTROL (Proportional)
   - Force proportional to error
   - Simple but often insufficient alone
   - Higher gain = faster but potentially unstable

3. PD-CONTROL (Proportional-Derivative)
   - Adds damping through velocity feedback
   - Reduces oscillations
   - Can stabilize many systems

4. PID-CONTROL (Proportional-Integral-Derivative)
   - Integral term eliminates steady-state error
   - Most common industrial controller
   - Requires careful tuning

5. FULL STATE FEEDBACK
   - Control all state variables
   - Enables multiple objectives (angle + position)
   - Requires more sensors or state estimation

6. DISTURBANCE REJECTION
   - Good controllers reject external forces
   - Integral action crucial for constant disturbances

7. NOISE SENSITIVITY
   - Real sensors have noise
   - Derivative terms amplify noise
   - May need filtering or observers

8. GAIN TUNING
   - Trade-offs: speed vs stability vs effort
   - Systematic methods available (LQR, pole placement)
""")

    def interactive_menu(self):
        """Interactive menu for running experiments."""
        while True:
            print("\n" + "="*50)
            print("TESTER AGENT - Interactive Menu")
            print("="*50)
            print("1. Open-Loop Behavior (no control)")
            print("2. P-Control (proportional only)")
            print("3. PD-Control (proportional + derivative)")
            print("4. PID-Control (full PID)")
            print("5. Full State Feedback (pole + cart position)")
            print("6. Disturbance Rejection")
            print("7. Noise Sensitivity")
            print("8. Gain Tuning Exploration")
            print("9. Run ALL experiments")
            print("0. Exit")
            print("-"*50)

            try:
                choice = input("Select experiment (0-9): ").strip()
            except EOFError:
                break

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                self.experiment_1_open_loop()
            elif choice == "2":
                self.experiment_2_p_control()
            elif choice == "3":
                self.experiment_3_pd_control()
            elif choice == "4":
                self.experiment_4_pid_control()
            elif choice == "5":
                self.experiment_5_cart_position_control()
            elif choice == "6":
                self.experiment_6_disturbance_rejection()
            elif choice == "7":
                self.experiment_7_noise_sensitivity()
            elif choice == "8":
                self.experiment_8_gain_tuning_exploration()
            elif choice == "9":
                self.run_all_experiments()
            else:
                print("Invalid choice. Please enter 0-9.")


def main():
    parser = argparse.ArgumentParser(
        description="Tester Agent for learning control systems with the inverted pendulum"
    )
    parser.add_argument("--experiment", "-e", type=int, choices=range(1, 9),
                       help="Run a specific experiment (1-8)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting (for headless environments)")

    args = parser.parse_args()

    agent = TesterAgent(verbose=not args.quiet, plot=not args.no_plot)

    if args.all:
        agent.run_all_experiments()
    elif args.experiment:
        experiments = {
            1: agent.experiment_1_open_loop,
            2: agent.experiment_2_p_control,
            3: agent.experiment_3_pd_control,
            4: agent.experiment_4_pid_control,
            5: agent.experiment_5_cart_position_control,
            6: agent.experiment_6_disturbance_rejection,
            7: agent.experiment_7_noise_sensitivity,
            8: agent.experiment_8_gain_tuning_exploration,
        }
        experiments[args.experiment]()
    else:
        agent.interactive_menu()


if __name__ == "__main__":
    main()
