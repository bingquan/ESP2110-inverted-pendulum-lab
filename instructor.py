#!/usr/bin/env python3
"""
Instructor Agent for ESP2110 Inverted Pendulum Lab

An agent that creates educational Jupyter notebooks to teach students
about control systems using the inverted pendulum cart as the running example.

Usage:
    python instructor.py                     # Interactive menu
    python instructor.py --topic <name>      # Generate specific topic notebook
    python instructor.py --list              # List available topics
    python instructor.py --all               # Generate all notebooks
    python instructor.py --output <dir>      # Specify output directory
"""

import json
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class NotebookBuilder:
    """Helper class to build Jupyter notebooks programmatically."""

    def __init__(self):
        self.cells: List[Dict[str, Any]] = []
        self.cell_counter = 0

    def add_markdown(self, content: str) -> 'NotebookBuilder':
        """Add a markdown cell."""
        self.cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": self._split_source(content),
            "id": f"cell-{self.cell_counter}"
        })
        self.cell_counter += 1
        return self

    def add_code(self, code: str, outputs: Optional[List] = None) -> 'NotebookBuilder':
        """Add a code cell."""
        self.cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": self._split_source(code),
            "id": f"cell-{self.cell_counter}"
        })
        self.cell_counter += 1
        return self

    def _split_source(self, content: str) -> List[str]:
        """Split content into lines for notebook format."""
        lines = content.split('\n')
        result = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                result.append(line + '\n')
            else:
                result.append(line)
        return result

    def build(self) -> Dict[str, Any]:
        """Build the complete notebook structure."""
        return {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }

    def save(self, filepath: str) -> None:
        """Save the notebook to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.build(), f, indent=1)


class InstructorAgent:
    """
    An educational agent that creates Jupyter notebooks to teach
    control systems concepts using the inverted pendulum.
    """

    def __init__(self, output_dir: str = "generated_notebooks", verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)

        # Available topics and their generators
        self.topics = {
            "intro_control": self.create_intro_control_notebook,
            "system_modeling": self.create_system_modeling_notebook,
            "linearization": self.create_linearization_notebook,
            "stability_analysis": self.create_stability_analysis_notebook,
            "p_control": self.create_p_control_notebook,
            "pd_control": self.create_pd_control_notebook,
            "pid_control": self.create_pid_control_notebook,
            "state_feedback": self.create_state_feedback_notebook,
            "pole_placement": self.create_pole_placement_notebook,
            "disturbance_rejection": self.create_disturbance_rejection_notebook,
            "observer_design": self.create_observer_design_notebook,
            "lqr_control": self.create_lqr_control_notebook,
        }

        self.topic_descriptions = {
            "intro_control": "Introduction to Control Systems",
            "system_modeling": "Mathematical Modeling of Dynamic Systems",
            "linearization": "Linearization Around Equilibrium Points",
            "stability_analysis": "Stability Analysis and Eigenvalues",
            "p_control": "Proportional (P) Control",
            "pd_control": "Proportional-Derivative (PD) Control",
            "pid_control": "PID Control and Tuning",
            "state_feedback": "Full State Feedback Control",
            "pole_placement": "Pole Placement Design",
            "disturbance_rejection": "Disturbance Rejection and Robustness",
            "observer_design": "State Observers and Estimation",
            "lqr_control": "Linear Quadratic Regulator (LQR)",
        }

    def log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def list_topics(self) -> None:
        """List all available topics."""
        print("\nAvailable Topics:")
        print("-" * 60)
        for i, (key, desc) in enumerate(self.topic_descriptions.items(), 1):
            print(f"  {i:2d}. {key:25s} - {desc}")
        print("-" * 60)

    def create_notebook(self, topic: str) -> Optional[str]:
        """Create a notebook for the specified topic."""
        if topic not in self.topics:
            self.log(f"Unknown topic: {topic}")
            return None

        generator = self.topics[topic]
        filepath = generator()
        return filepath

    def create_all_notebooks(self) -> List[str]:
        """Create all available notebooks."""
        created = []
        for topic in self.topics:
            filepath = self.create_notebook(topic)
            if filepath:
                created.append(filepath)
        return created

    # =========================================================================
    # COMMON NOTEBOOK COMPONENTS
    # =========================================================================

    def _add_header(self, nb: NotebookBuilder, title: str, objectives: List[str],
                    prereqs: Optional[List[str]] = None) -> None:
        """Add standard header to notebook."""
        header = f"""# {title}

This notebook is designed to run in Google Colab or local Jupyter.

**Colab steps:** Open the notebook, run the setup cell below, then run cells top-to-bottom.

## Learning Objectives
"""
        for obj in objectives:
            header += f"- {obj}\n"

        if prereqs:
            header += "\n## Prerequisites\n"
            for prereq in prereqs:
                header += f"- {prereq}\n"

        header += """
---
"""
        nb.add_markdown(header)

    def _add_setup_cell(self, nb: NotebookBuilder) -> None:
        """Add the standard setup/import cell."""
        nb.add_code("""# Install and import required packages
# Uncomment the next line if running in Colab
# !pip -q install numpy scipy matplotlib control

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint, solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

print("Setup complete!")""")

    def _add_pendulum_class(self, nb: NotebookBuilder, include_friction: bool = False) -> None:
        """Add the InvertedPendulum class definition."""
        if include_friction:
            code = '''class InvertedPendulum:
    """Inverted Pendulum on a Cart with friction."""

    def __init__(self, dt=0.02):
        # System parameters
        self.g = 9.8          # Gravity (m/s^2)
        self.m_c = 1.0        # Cart mass (kg)
        self.m_p = 0.1        # Pole mass (kg)
        self.L = 0.5          # Pole length (m)
        self.dt = dt          # Time step (s)
        self.b_c = 0.1        # Cart friction coefficient
        self.b_p = 0.01       # Pole damping coefficient

        # State: [x, x_dot, theta, theta_dot]
        self.state = np.zeros(4)

    def reset(self, state=None):
        """Reset to initial state."""
        if state is not None:
            self.state = np.array(state, dtype=float)
        else:
            self.state = np.random.uniform(-0.05, 0.05, size=4)
        return self.state.copy()

    def dynamics(self, state, force):
        """Compute state derivatives (nonlinear dynamics with friction)."""
        x, x_dot, theta, theta_dot = state

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        # Effective force after cart friction
        F_eff = force - self.b_c * x_dot

        # Pole damping torque
        tau_damp = -self.b_p * theta_dot

        # Denominator term
        denom = self.m_c + self.m_p * sin_t**2

        # Accelerations
        x_ddot = (F_eff + self.m_p * sin_t * (self.L * theta_dot**2 - self.g * cos_t)) / denom
        theta_ddot = (-F_eff * cos_t - self.m_p * self.L * theta_dot**2 * sin_t * cos_t
                      + (self.m_c + self.m_p) * self.g * sin_t + tau_damp / self.L) / (self.L * denom)

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, force):
        """Advance one time step using RK4 integration."""
        k1 = self.dynamics(self.state, force)
        k2 = self.dynamics(self.state + 0.5 * self.dt * k1, force)
        k3 = self.dynamics(self.state + 0.5 * self.dt * k2, force)
        k4 = self.dynamics(self.state + self.dt * k3, force)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state.copy()

# Create pendulum instance
pendulum = InvertedPendulum(dt=0.02)
print(f"Pendulum parameters: m_c={pendulum.m_c}kg, m_p={pendulum.m_p}kg, L={pendulum.L}m")'''
        else:
            code = '''class InvertedPendulum:
    """Inverted Pendulum on a Cart (simplified, no friction)."""

    def __init__(self, dt=0.02):
        # System parameters
        self.g = 9.8          # Gravity (m/s^2)
        self.m_c = 1.0        # Cart mass (kg)
        self.m_p = 0.1        # Pole mass (kg)
        self.L = 0.5          # Pole length (m)
        self.dt = dt          # Time step (s)

        # State: [x, x_dot, theta, theta_dot]
        self.state = np.zeros(4)

    def reset(self, state=None):
        """Reset to initial state."""
        if state is not None:
            self.state = np.array(state, dtype=float)
        else:
            self.state = np.random.uniform(-0.05, 0.05, size=4)
        return self.state.copy()

    def dynamics(self, state, force):
        """Compute state derivatives (nonlinear dynamics)."""
        x, x_dot, theta, theta_dot = state

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        denom = self.m_c + self.m_p * sin_t**2

        x_ddot = (force + self.m_p * sin_t * (self.L * theta_dot**2 - self.g * cos_t)) / denom
        theta_ddot = (-force * cos_t - self.m_p * self.L * theta_dot**2 * sin_t * cos_t
                      + (self.m_c + self.m_p) * self.g * sin_t) / (self.L * denom)

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, force):
        """Advance one time step using RK4 integration."""
        k1 = self.dynamics(self.state, force)
        k2 = self.dynamics(self.state + 0.5 * self.dt * k1, force)
        k3 = self.dynamics(self.state + 0.5 * self.dt * k2, force)
        k4 = self.dynamics(self.state + self.dt * k3, force)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state.copy()

# Create pendulum instance
pendulum = InvertedPendulum(dt=0.02)
print(f"Pendulum parameters: m_c={pendulum.m_c}kg, m_p={pendulum.m_p}kg, L={pendulum.L}m")'''
        nb.add_code(code)

    def _add_simulation_helper(self, nb: NotebookBuilder) -> None:
        """Add simulation and plotting helper functions."""
        nb.add_code('''def simulate(pendulum, controller, steps, initial_state=None, disturbance=None):
    """
    Run a closed-loop simulation.

    Args:
        pendulum: InvertedPendulum instance
        controller: function(state, t) -> force
        steps: number of simulation steps
        initial_state: optional initial state [x, x_dot, theta, theta_dot]
        disturbance: optional function(t) -> disturbance force

    Returns:
        dict with time, states, and forces arrays
    """
    if initial_state is not None:
        pendulum.reset(initial_state)
    else:
        pendulum.reset()

    dt = pendulum.dt
    states = [pendulum.state.copy()]
    forces = []
    times = [0.0]

    for t in range(steps):
        force = controller(pendulum.state, t * dt)
        if disturbance is not None:
            force += disturbance(t * dt)
        forces.append(force)
        state = pendulum.step(force)
        states.append(state.copy())
        times.append((t + 1) * dt)

    return {
        'time': np.array(times[:-1]),
        'states': np.array(states[:-1]),
        'forces': np.array(forces)
    }


def plot_response(result, title="System Response", theta_ref=np.pi):
    """Plot simulation results."""
    t = result['time']
    x, x_dot, theta, theta_dot = result['states'].T
    forces = result['forces']

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Cart position
    axes[0, 0].plot(t, x, 'b-')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Cart velocity
    axes[0, 1].plot(t, x_dot, 'b-')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Cart Velocity')

    # Pole angle
    axes[1, 0].plot(t, np.degrees(theta), 'r-')
    axes[1, 0].axhline(y=np.degrees(theta_ref), color='g', linestyle='--', label='Reference')
    axes[1, 0].set_ylabel('Angle (deg)')
    axes[1, 0].set_title('Pole Angle')
    axes[1, 0].legend()

    # Angular velocity
    axes[1, 1].plot(t, np.degrees(theta_dot), 'r-')
    axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
    axes[1, 1].set_title('Pole Angular Velocity')

    # Control force
    axes[2, 0].plot(t, forces, 'g-')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Force (N)')
    axes[2, 0].set_title('Control Force')

    # Angle error
    error = np.degrees(theta - theta_ref)
    axes[2, 1].plot(t, error, 'm-')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Error (deg)')
    axes[2, 1].set_title('Angle Error')
    axes[2, 1].axhline(y=0, color='g', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"Final angle error: {error[-1]:.4f} deg")
    print(f"RMS angle error: {np.sqrt(np.mean(error**2)):.4f} deg")
    print(f"Max control force: {np.max(np.abs(forces)):.2f} N")


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

print("Simulation helpers loaded!")''')

    def _add_footer(self, nb: NotebookBuilder, checkpoints: List[str],
                    pitfalls: List[str], next_topic: Optional[str] = None) -> None:
        """Add standard footer to notebook."""
        footer = """---
## Checkpoints

"""
        for cp in checkpoints:
            footer += f"- [ ] {cp}\n"

        footer += """
## Common Pitfalls

"""
        for pitfall in pitfalls:
            footer += f"- {pitfall}\n"

        if next_topic:
            footer += f"""
---
## Next Steps

Continue to the next notebook: **{next_topic}**
"""

        footer += f"""
---
*Generated by Instructor Agent on {datetime.now().strftime('%Y-%m-%d')}*
"""
        nb.add_markdown(footer)

    # =========================================================================
    # NOTEBOOK GENERATORS
    # =========================================================================

    def create_intro_control_notebook(self) -> str:
        """Create Introduction to Control Systems notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Introduction to Control Systems",
            objectives=[
                "Define what a control system is and why it's needed",
                "Distinguish between open-loop and closed-loop control",
                "Identify the components of a feedback control system",
                "Understand the inverted pendulum as a control problem"
            ]
        )

        self._add_setup_cell(nb)

        nb.add_markdown("""## What is Control Systems Engineering?

**Control systems engineering** is the discipline that applies control theory to design systems with predictable behavior.

### Key Questions Control Engineers Ask:
1. How does a system behave naturally (dynamics)?
2. How can we make it behave the way we want (control)?
3. How do we ensure it remains stable and robust?

### Real-World Examples:
- **Cruise control**: Maintains car speed despite hills
- **Thermostat**: Regulates room temperature
- **Autopilot**: Stabilizes and guides aircraft
- **Robotic arms**: Precise positioning and movement
""")

        nb.add_markdown("""## Open-Loop vs Closed-Loop Control

### Open-Loop Control
- No feedback from the output
- Control action is predetermined
- Cannot correct for disturbances
- Example: A toaster with a timer

### Closed-Loop (Feedback) Control
- Output is measured and fed back
- Control action depends on error
- Can reject disturbances
- Example: A thermostat

```
                    ┌─────────────┐
Reference  ──────►  │  Controller │  ────►  System  ────► Output
    +               └─────────────┘                         │
    │                                                       │
    └───────────────── Feedback ◄───────────────────────────┘
                         (-)
```
""")

        nb.add_code('''# Demonstration: Open-loop vs Closed-loop

# Simple first-order system: dx/dt = -a*x + b*u
def simulate_first_order(controller, a=1.0, b=1.0, x0=0.0, setpoint=1.0,
                          steps=200, dt=0.05, disturbance=0.0):
    x = x0
    history = {'t': [], 'x': [], 'u': [], 'e': []}

    for i in range(steps):
        t = i * dt
        error = setpoint - x
        u = controller(error, t)

        # Apply disturbance halfway through
        d = disturbance if t > 5.0 else 0.0

        # Euler integration
        x = x + dt * (-a * x + b * u + d)

        history['t'].append(t)
        history['x'].append(x)
        history['u'].append(u)
        history['e'].append(error)

    return {k: np.array(v) for k, v in history.items()}

# Open-loop controller: fixed input
def open_loop_controller(error, t):
    return 1.0  # Constant input

# Closed-loop (proportional) controller
def closed_loop_controller(error, t):
    return 2.0 * error  # Proportional gain

# Run simulations
result_open = simulate_first_order(open_loop_controller, disturbance=0.5)
result_closed = simulate_first_order(closed_loop_controller, disturbance=0.5)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(result_open['t'], result_open['x'], 'b-', label='Open-loop')
axes[0].plot(result_closed['t'], result_closed['x'], 'r-', label='Closed-loop')
axes[0].axhline(y=1.0, color='g', linestyle='--', label='Setpoint')
axes[0].axvline(x=5.0, color='gray', linestyle=':', label='Disturbance starts')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Output x')
axes[0].set_title('Response with Disturbance')
axes[0].legend()

axes[1].plot(result_open['t'], result_open['e'], 'b-', label='Open-loop')
axes[1].plot(result_closed['t'], result_closed['e'], 'r-', label='Closed-loop')
axes[1].axhline(y=0, color='g', linestyle='--')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Error')
axes[1].set_title('Tracking Error')
axes[1].legend()

plt.tight_layout()
plt.show()

print("Notice how closed-loop control rejects the disturbance!")''')

        nb.add_markdown(r"""## The Inverted Pendulum Problem

The **inverted pendulum on a cart** is a classic control problem because:

1. **It's naturally unstable** - the upright position is an unstable equilibrium
2. **It's underactuated** - we can only push the cart, not directly control the pole
3. **It demonstrates key concepts** - stability, feedback, robustness
4. **It has real applications** - balancing robots, rocket landing, Segways

### The Challenge
Keep the pole balanced upright by moving the cart left and right!

### State Variables
| Variable | Symbol | Description |
|----------|--------|-------------|
| Cart position | $x$ | Horizontal position of cart (m) |
| Cart velocity | $\dot{x}$ | Rate of change of position (m/s) |
| Pole angle | $\\theta$ | Angle from vertical (rad) |
| Angular velocity | $\dot{\\theta}$ | Rate of angle change (rad/s) |

### Control Input
- **Force $F$**: Applied horizontally to the cart (N)
""")

        self._add_pendulum_class(nb)

        nb.add_code('''# Visualize the unstable nature of the inverted pendulum

def plot_pendulum_phase(pendulum, title="Phase Portrait"):
    """Plot how different initial angles evolve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    angles = [175, 178, 180, 182, 185]  # degrees
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    for angle, color in zip(angles, colors):
        pendulum.reset([0, 0, np.radians(angle), 0])
        states = [pendulum.state.copy()]

        for _ in range(200):
            pendulum.step(0)  # No control
            states.append(pendulum.state.copy())

        states = np.array(states)
        axes[0].plot(np.arange(len(states)) * pendulum.dt,
                     np.degrees(states[:, 2]), color=color,
                     label=f'{angle}°')
        axes[1].plot(np.degrees(states[:, 2]),
                     np.degrees(states[:, 3]), color=color)

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pole Angle (deg)')
    axes[0].set_title('Angle vs Time (No Control)')
    axes[0].axhline(y=180, color='k', linestyle='--', alpha=0.5)
    axes[0].legend()

    axes[1].set_xlabel('Angle (deg)')
    axes[1].set_ylabel('Angular Velocity (deg/s)')
    axes[1].set_title('Phase Portrait')
    axes[1].axvline(x=180, color='k', linestyle='--', alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

pendulum = InvertedPendulum(dt=0.02)
plot_pendulum_phase(pendulum, "Inverted Pendulum: Unstable Without Control")

print("\\nObservation: Even tiny deviations from 180° grow exponentially!")
print("This is the defining characteristic of an UNSTABLE equilibrium.")''')

        nb.add_markdown("""## Task: Explore the System (15 mins)

1. **Vary initial conditions**: Try different starting angles near 180° and observe how quickly the pole falls.

2. **Apply constant force**: What happens if you apply a constant force to the cart?

3. **Try random forces**: Can random inputs stabilize the system?

**Questions to answer:**
- Why can't we simply apply a constant force to balance the pole?
- What information would a controller need to stabilize the system?
- Why is feedback essential for this problem?
""")

        nb.add_code('''# TODO: Experiment with different scenarios

pendulum = InvertedPendulum(dt=0.02)

# Scenario 1: Start at exactly 180 degrees
pendulum.reset([0, 0, np.pi, 0])
print(f"Starting at exactly 180°: {np.degrees(pendulum.state[2]):.2f}°")

# Simulate for 100 steps with no control
for _ in range(100):
    pendulum.step(0)
print(f"After 2 seconds (no control): {np.degrees(pendulum.state[2]):.2f}°")
print("Numerically, perfect 180° is stable, but any perturbation causes divergence!")

# Scenario 2: Constant force
pendulum.reset([0, 0, np.pi + 0.01, 0])
for _ in range(100):
    pendulum.step(5.0)  # Constant rightward force
print(f"\\nWith constant force: Final angle = {np.degrees(pendulum.state[2]):.2f}°")

# Scenario 3: Random forces
pendulum.reset([0, 0, np.pi + 0.01, 0])
for _ in range(100):
    pendulum.step(np.random.uniform(-10, 10))
print(f"With random forces: Final angle = {np.degrees(pendulum.state[2]):.2f}°")

print("\\nConclusion: We need INTELLIGENT feedback control!")''')

        self._add_footer(nb,
            checkpoints=[
                "Can explain the difference between open-loop and closed-loop control",
                "Understand why the inverted pendulum is unstable",
                "Can identify the four state variables of the cart-pole system",
                "Ran simulations showing the need for feedback control"
            ],
            pitfalls=[
                "Confusing stability with equilibrium - 180° is an equilibrium but unstable",
                "Thinking open-loop control can work for unstable systems",
                "Forgetting that angles are in radians in the code (180° = π rad)"
            ],
            next_topic="System Modeling"
        )

        filepath = os.path.join(self.output_dir, "01_intro_control.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_system_modeling_notebook(self) -> str:
        """Create System Modeling notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Mathematical Modeling of Dynamic Systems",
            objectives=[
                "Derive equations of motion using Newton's laws",
                "Express dynamics in state-space form",
                "Understand the relationship between physical parameters and system behavior",
                "Simulate nonlinear system dynamics"
            ],
            prereqs=["Introduction to Control Systems"]
        )

        self._add_setup_cell(nb)

        nb.add_markdown("""## Deriving the Equations of Motion

We use **Newton's second law** (F = ma) to derive the dynamics of the cart-pole system.

### Free Body Diagram

Consider forces acting on the cart and pole:

**Cart:**
- Applied force: $F$ (control input)
- Reaction from pole: $H$ (horizontal), $V$ (vertical)

**Pole:**
- Weight: $m_p g$ (at center of mass)
- Reaction forces: $-H$, $-V$
- Moment about pivot

### Assumptions
1. Pole is a uniform rod (center of mass at L/2)
2. Frictionless pivot and track (for now)
3. Pole mass concentrated at the end (for simplified model)
""")

        nb.add_markdown(r"""## The Equations of Motion

After applying Newton's laws and simplifying, we get:

### Cart acceleration:
$$\ddot{x} = \frac{F + m_p \sin\theta (L \dot{\theta}^2 - g \cos\theta)}{m_c + m_p \sin^2\theta}$$

### Pole angular acceleration:
$$\ddot{\theta} = \frac{-F \cos\theta - m_p L \dot{\theta}^2 \sin\theta \cos\theta + (m_c + m_p) g \sin\theta}{L(m_c + m_p \sin^2\theta)}$$

These are **nonlinear differential equations** because of the $\sin\theta$, $\cos\theta$, and $\dot{\theta}^2$ terms.

### State-Space Form

We can write this as a first-order system with state $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T$:

$$\dot{\mathbf{x}} = f(\mathbf{x}, u)$$

where $u = F$ is the control input.
""")

        self._add_pendulum_class(nb)

        nb.add_code('''# Verify the dynamics implementation

def test_dynamics():
    """Test that the dynamics are correctly implemented."""
    pend = InvertedPendulum(dt=0.01)

    # Test 1: At rest, hanging down (theta=0), should stay at rest
    pend.reset([0, 0, 0, 0])
    for _ in range(100):
        pend.step(0)
    print(f"Test 1 - Hanging down, no force:")
    print(f"  theta should be ~0: {pend.state[2]:.6f} rad")

    # Test 2: Apply force, cart should accelerate
    pend.reset([0, 0, 0, 0])
    pend.step(10)  # Apply 10N force
    print(f"\\nTest 2 - Apply 10N force:")
    print(f"  Cart velocity should be positive: {pend.state[1]:.4f} m/s")

    # Test 3: Near upright, small perturbation should grow
    pend.reset([0, 0, np.pi - 0.01, 0])
    for _ in range(50):
        pend.step(0)
    print(f"\\nTest 3 - Near upright (179.4°), no force:")
    print(f"  Angle deviation grows: {np.degrees(pend.state[2]):.2f}°")

test_dynamics()''')

        nb.add_markdown("""## Effect of System Parameters

The behavior of the system depends on:
- **Cart mass** ($m_c$): Higher mass = slower response
- **Pole mass** ($m_p$): Affects coupling between cart and pole
- **Pole length** ($L$): Longer pole = slower fall, easier to balance
- **Gravity** ($g$): Higher = faster fall

Let's explore these effects!
""")

        nb.add_code('''# Parameter sensitivity analysis

def compare_parameters(param_name, values, default_state=[0, 0, np.pi + 0.05, 0]):
    """Compare system behavior for different parameter values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))

    for val, color in zip(values, colors):
        pend = InvertedPendulum(dt=0.01)

        # Set parameter
        if param_name == 'm_c':
            pend.m_c = val
        elif param_name == 'm_p':
            pend.m_p = val
        elif param_name == 'L':
            pend.L = val

        pend.reset(default_state.copy())
        states = [pend.state.copy()]

        for _ in range(300):
            pend.step(0)  # No control
            states.append(pend.state.copy())
            if abs(states[-1][2] - np.pi) > 1.5:  # Stop if fallen too far
                break

        states = np.array(states)
        t = np.arange(len(states)) * pend.dt

        axes[0].plot(t, np.degrees(states[:, 2]), color=color, label=f'{param_name}={val}')
        axes[1].plot(t, states[:, 0], color=color, label=f'{param_name}={val}')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pole Angle (deg)')
    axes[0].set_title('Pole Angle Evolution')
    axes[0].axhline(y=180, color='k', linestyle='--', alpha=0.5)
    axes[0].legend()

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Cart Position (m)')
    axes[1].set_title('Cart Position')
    axes[1].legend()

    plt.suptitle(f'Effect of {param_name}')
    plt.tight_layout()
    plt.show()

# Compare different pole lengths
compare_parameters('L', [0.3, 0.5, 0.7, 1.0])

# Compare different cart masses
compare_parameters('m_c', [0.5, 1.0, 2.0, 5.0])

print("Observation: Longer poles fall slower (more time to react)!")
print("Heavier carts are harder to move but the pole dynamics are similar.")''')

        nb.add_markdown("""## Task: Derive the Linearized Model (20 mins)

For small angles near the upright position ($\\theta \\approx \\pi$), we can linearize:
- $\\sin(\\theta - \\pi) \\approx \\theta - \\pi = \\phi$ (small angle)
- $\\cos(\\theta - \\pi) \\approx -1$
- $\\dot{\\theta}^2 \\approx 0$ (small velocities)

**Exercise:**
1. Let $\\phi = \\theta - \\pi$ (deviation from upright)
2. Substitute the approximations into the equations of motion
3. Derive the linearized state-space matrices A and B

The result should be:
$$\\dot{\\mathbf{x}} = A\\mathbf{x} + Bu$$

where $\\mathbf{x} = [x, \\dot{x}, \\phi, \\dot{\\phi}]^T$
""")

        nb.add_code('''# TODO: Complete the linearization

def get_linearized_matrices(m_c, m_p, L, g):
    """
    Return the A and B matrices for the linearized system around theta = pi.

    State: [x, x_dot, phi, phi_dot] where phi = theta - pi (deviation from upright)
    Input: F (force on cart)
    """
    # Total mass
    M = m_c + m_p

    # Linearized A matrix
    # TODO: Fill in the correct values based on your derivation
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, m_p * g / m_c, 0],           # Approximate: assumes m_p << m_c
        [0, 0, 0, 1],
        [0, 0, (M * g) / (m_c * L), 0]      # Approximate
    ])

    # Linearized B matrix
    B = np.array([
        [0],
        [1 / m_c],
        [0],
        [1 / (m_c * L)]
    ])

    return A, B

# Test the linearization
pend = InvertedPendulum()
A, B = get_linearized_matrices(pend.m_c, pend.m_p, pend.L, pend.g)

print("Linearized A matrix:")
print(A)
print("\\nLinearized B matrix:")
print(B)

# Check eigenvalues (stability)
eigenvalues = np.linalg.eigvals(A)
print("\\nEigenvalues of A:")
for ev in eigenvalues:
    print(f"  {ev:.4f} (stable: {ev.real < 0})")''')

        self._add_footer(nb,
            checkpoints=[
                "Can write the nonlinear equations of motion",
                "Understand how physical parameters affect dynamics",
                "Successfully ran parameter sensitivity simulations",
                "Started working on the linearized model"
            ],
            pitfalls=[
                "Forgetting that theta=0 is hanging down, theta=pi is upright",
                "Sign errors in the equations of motion",
                "Not accounting for the moment of inertia of the pole"
            ],
            next_topic="Linearization"
        )

        filepath = os.path.join(self.output_dir, "02_system_modeling.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_linearization_notebook(self) -> str:
        """Create Linearization notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Linearization Around Equilibrium Points",
            objectives=[
                "Understand the concept of equilibrium points",
                "Apply Taylor series expansion for linearization",
                "Derive the linearized state-space model",
                "Compare linear and nonlinear model predictions"
            ],
            prereqs=["System Modeling"]
        )

        self._add_setup_cell(nb)
        self._add_pendulum_class(nb)

        nb.add_markdown(r"""## Equilibrium Points

An **equilibrium point** is where the system can stay indefinitely without input:
$$f(\mathbf{x}_{eq}, 0) = 0$$

For the inverted pendulum, there are two equilibria:
1. **Hanging down** ($\theta = 0$): Stable - the pole naturally stays here
2. **Upright** ($\theta = \pi$): Unstable - any perturbation causes the pole to fall

### Linearization Strategy

Near an equilibrium, nonlinear dynamics can be approximated by a linear system:
$$\dot{\mathbf{x}} = A\mathbf{x} + Bu$$

where:
- $A = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_{eq}}$ (Jacobian)
- $B = \frac{\partial f}{\partial u}\bigg|_{\mathbf{x}_{eq}}$
""")

        nb.add_code('''def compute_jacobians(pendulum, x_eq, u_eq=0, epsilon=1e-6):
    """
    Numerically compute the A and B matrices via finite differences.
    """
    n_states = 4
    n_inputs = 1

    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_inputs))

    # Compute A matrix (df/dx)
    for i in range(n_states):
        x_plus = x_eq.copy()
        x_minus = x_eq.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = pendulum.dynamics(x_plus, u_eq)
        f_minus = pendulum.dynamics(x_minus, u_eq)

        A[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # Compute B matrix (df/du)
    f_plus = pendulum.dynamics(x_eq, u_eq + epsilon)
    f_minus = pendulum.dynamics(x_eq, u_eq - epsilon)
    B[:, 0] = (f_plus - f_minus) / (2 * epsilon)

    return A, B

# Compute linearization at upright equilibrium
pendulum = InvertedPendulum()
x_upright = np.array([0, 0, np.pi, 0])

A_upright, B_upright = compute_jacobians(pendulum, x_upright)

print("Linearization at UPRIGHT equilibrium (theta = pi):")
print("\\nA matrix:")
print(np.round(A_upright, 4))
print("\\nB matrix:")
print(np.round(B_upright, 4))

# Also compute for hanging equilibrium
x_hanging = np.array([0, 0, 0, 0])
A_hanging, B_hanging = compute_jacobians(pendulum, x_hanging)

print("\\n" + "="*50)
print("Linearization at HANGING equilibrium (theta = 0):")
print("\\nA matrix:")
print(np.round(A_hanging, 4))''')

        nb.add_markdown("""## Analytical Linearization

For the inverted pendulum at $\\theta = \\pi$, let $\\phi = \\theta - \\pi$ (deviation from upright).

Using small-angle approximations:
- $\\sin(\\phi) \\approx \\phi$
- $\\cos(\\phi) \\approx 1$
- $\\dot{\\phi}^2 \\approx 0$

The linearized matrices are:
""")

        nb.add_code('''def analytical_linearization(m_c, m_p, L, g):
    """
    Analytically derived linearization at theta = pi.

    State: [x, x_dot, phi, phi_dot] where phi = theta - pi
    """
    # For the simplified model (pole mass at end)
    denom = m_c + m_p

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, (m_p * g) / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (denom * g) / (L * m_c), 0]
    ])

    B = np.array([
        [0],
        [1 / denom],
        [0],
        [1 / (L * m_c)]
    ])

    return A, B

# Compare numerical and analytical linearization
pend = InvertedPendulum()
A_num, B_num = compute_jacobians(pend, np.array([0, 0, np.pi, 0]))
A_ana, B_ana = analytical_linearization(pend.m_c, pend.m_p, pend.L, pend.g)

print("Numerical vs Analytical Comparison:")
print("\\nA matrix difference (should be small):")
print(np.round(A_num - A_ana, 6))
print("\\nB matrix difference:")
print(np.round(B_num - B_ana, 6))

# Note: There might be small differences due to the specific model formulation''')

        nb.add_markdown("""## Comparing Linear and Nonlinear Models

The linearization is only accurate near the equilibrium point. Let's see how well it predicts!
""")

        nb.add_code('''def compare_linear_nonlinear(pendulum, A, B, initial_deviation, steps=200):
    """Compare nonlinear simulation with linear prediction."""
    dt = pendulum.dt

    # Initial state (small deviation from upright)
    x0 = np.array([0, 0, np.pi + initial_deviation, 0])

    # Nonlinear simulation
    pendulum.reset(x0.copy())
    nonlinear_states = [pendulum.state.copy()]
    for _ in range(steps):
        pendulum.step(0)  # No control
        nonlinear_states.append(pendulum.state.copy())
    nonlinear_states = np.array(nonlinear_states)

    # Linear simulation (using deviation coordinates)
    # phi = theta - pi
    x_lin = np.array([0, 0, initial_deviation, 0])  # Deviation from upright
    linear_states = [x_lin.copy()]
    for _ in range(steps):
        x_dot = A @ x_lin + B.flatten() * 0  # No control
        x_lin = x_lin + dt * x_dot
        linear_states.append(x_lin.copy())
    linear_states = np.array(linear_states)

    # Convert linear back to absolute coordinates for comparison
    linear_states[:, 2] += np.pi  # phi + pi = theta

    # Plot comparison
    t = np.arange(len(nonlinear_states)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['Cart Position (m)', 'Cart Velocity (m/s)',
              'Pole Angle (deg)', 'Angular Velocity (deg/s)']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        if i >= 2:  # Angular quantities in degrees
            ax.plot(t, np.degrees(nonlinear_states[:, i]), 'b-',
                   label='Nonlinear', linewidth=2)
            ax.plot(t, np.degrees(linear_states[:, i]), 'r--',
                   label='Linear', linewidth=2)
        else:
            ax.plot(t, nonlinear_states[:, i], 'b-',
                   label='Nonlinear', linewidth=2)
            ax.plot(t, linear_states[:, i], 'r--',
                   label='Linear', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend()
        ax.set_title(label)

    plt.suptitle(f'Linear vs Nonlinear (initial deviation = {np.degrees(initial_deviation):.1f}°)')
    plt.tight_layout()
    plt.show()

# Compare for different initial deviations
pend = InvertedPendulum(dt=0.01)
A, B = compute_jacobians(pend, np.array([0, 0, np.pi, 0]))

print("Small deviation (1°) - Linear model should be accurate:")
compare_linear_nonlinear(pend, A, B, np.radians(1))

print("\\nLarger deviation (10°) - Linear model starts to diverge:")
compare_linear_nonlinear(pend, A, B, np.radians(10))''')

        nb.add_markdown("""## Task: Validity of Linearization (15 mins)

**Questions to investigate:**

1. At what angle deviation does the linear model become inaccurate?
2. How does the error grow over time?
3. Why does the linear model predict exponential growth?

**Hint:** The eigenvalues of the A matrix tell us about the system's natural behavior!
""")

        nb.add_code('''# TODO: Investigate the validity range of linearization

# Analyze eigenvalues
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of A matrix:")
for i, ev in enumerate(eigenvalues):
    print(f"  lambda_{i+1} = {ev.real:+.4f} + {ev.imag:.4f}j")
    if ev.real > 0:
        print(f"    -> Unstable mode! Time constant = {1/ev.real:.3f} s")

# Your experiments here:
# 1. Try different initial angles
# 2. Measure when linear and nonlinear diverge by more than 10%
# 3. Plot the error vs initial angle''')

        self._add_footer(nb,
            checkpoints=[
                "Can identify equilibrium points of the cart-pole system",
                "Understand how to compute Jacobian matrices",
                "Successfully compared linear and nonlinear predictions",
                "Know the validity limits of the linear approximation"
            ],
            pitfalls=[
                "Forgetting to convert between deviation (phi) and absolute (theta) coordinates",
                "Using linearization too far from the equilibrium point",
                "Sign errors when linearizing around theta=pi vs theta=0"
            ],
            next_topic="Stability Analysis"
        )

        filepath = os.path.join(self.output_dir, "03_linearization.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_stability_analysis_notebook(self) -> str:
        """Create Stability Analysis notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Stability Analysis and Eigenvalues",
            objectives=[
                "Understand the concept of stability in control systems",
                "Analyze stability using eigenvalues of the A matrix",
                "Relate pole locations to system behavior",
                "Determine controllability of the system"
            ],
            prereqs=["Linearization"]
        )

        self._add_setup_cell(nb)

        nb.add_markdown(r"""## What is Stability?

A system is **stable** if, when perturbed from equilibrium, it returns to (or stays near) that equilibrium.

### Types of Stability:
1. **Asymptotically Stable**: Returns to equilibrium
2. **Marginally Stable**: Bounded but doesn't converge
3. **Unstable**: Diverges from equilibrium

### Mathematical Criterion

For the linear system $\dot{\mathbf{x}} = A\mathbf{x}$:

- **Stable** if all eigenvalues of $A$ have **negative real parts**
- **Unstable** if any eigenvalue has a **positive real part**

The eigenvalues are called the **poles** of the system.
""")

        nb.add_code('''# Stability analysis of the inverted pendulum

def analyze_stability(A, system_name="System"):
    """Analyze the stability of a linear system."""
    eigenvalues = np.linalg.eigvals(A)

    print(f"\\n{'='*50}")
    print(f"Stability Analysis: {system_name}")
    print('='*50)

    stable = True
    for i, ev in enumerate(eigenvalues):
        re = ev.real
        im = ev.imag

        status = "STABLE" if re < 0 else ("MARGINAL" if re == 0 else "UNSTABLE")
        if re >= 0:
            stable = False

        if im == 0:
            print(f"  Pole {i+1}: {re:+.4f} [{status}]")
            if re != 0:
                print(f"          Time constant: {abs(1/re):.3f} s")
        else:
            print(f"  Pole {i+1}: {re:+.4f} ± {abs(im):.4f}j [{status}]")
            if re != 0:
                print(f"          Damping: {-re/abs(ev):.3f}, Frequency: {abs(im)/(2*np.pi):.3f} Hz")

    print(f"\\nOverall: {'STABLE' if stable else 'UNSTABLE'}")
    return eigenvalues

# Linearization at upright (theta = pi)
def get_A_matrix(m_c=1.0, m_p=0.1, L=0.5, g=9.8):
    denom = m_c + m_p
    return np.array([
        [0, 1, 0, 0],
        [0, 0, (m_p * g) / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (denom * g) / (L * m_c), 0]
    ])

A_upright = get_A_matrix()
eigenvalues = analyze_stability(A_upright, "Inverted Pendulum (Upright)")''')

        nb.add_markdown("""## Pole-Zero Map

A **pole-zero map** visualizes eigenvalue locations in the complex plane.

- **Left half-plane** (Re < 0): Stable poles
- **Right half-plane** (Re > 0): Unstable poles
- **Imaginary axis** (Re = 0): Marginally stable
""")

        nb.add_code('''def plot_poles(eigenvalues, title="Pole-Zero Map"):
    """Plot poles in the complex plane."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot stability regions
    ax.axvspan(-10, 0, alpha=0.1, color='green', label='Stable Region')
    ax.axvspan(0, 10, alpha=0.1, color='red', label='Unstable Region')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=1)

    # Plot poles
    for ev in eigenvalues:
        color = 'green' if ev.real < 0 else 'red'
        ax.plot(ev.real, ev.imag, 'x', color=color, markersize=15,
               markeredgewidth=3)

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

plot_poles(eigenvalues, "Inverted Pendulum Poles (Uncontrolled)")

print("\\nThe positive real eigenvalue indicates exponential divergence!")
print("This is why the pendulum falls when not controlled.")''')

        nb.add_markdown(r"""## Controllability

A system is **controllable** if we can move from any initial state to any final state using the control input.

### Controllability Matrix
$$\mathcal{C} = [B \quad AB \quad A^2B \quad \cdots \quad A^{n-1}B]$$

The system is controllable if $\mathcal{C}$ has **full rank** (rank = n).
""")

        nb.add_code('''def check_controllability(A, B):
    """Check if the system is controllable."""
    n = A.shape[0]
    C = B.copy()

    for i in range(1, n):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])

    rank = np.linalg.matrix_rank(C)

    print("Controllability Matrix:")
    print(np.round(C, 4))
    print(f"\\nRank: {rank} (need {n} for controllability)")
    print(f"System is {'CONTROLLABLE' if rank == n else 'NOT CONTROLLABLE'}")

    return rank == n

# Check controllability
B = np.array([[0], [1/1.1], [0], [1/(0.5*1.0)]])  # From linearization
is_controllable = check_controllability(A_upright, B)

print("\\nGood news: The system is controllable!")
print("This means we CAN stabilize the pendulum with proper control.")''')

        nb.add_markdown("""## Task: Analyze Different Configurations (20 mins)

How do the poles change with different system parameters?

1. Vary the pole length (L): Does a longer pole have slower dynamics?
2. Vary the cart mass (m_c): How does this affect stability?
3. What parameter combination gives the "fastest" unstable pole?
""")

        nb.add_code('''# TODO: Parameter study

def parameter_study(param_name, values):
    """Study how poles change with a parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_poles = []
    for val in values:
        if param_name == 'L':
            A = get_A_matrix(L=val)
        elif param_name == 'm_c':
            A = get_A_matrix(m_c=val)
        elif param_name == 'm_p':
            A = get_A_matrix(m_p=val)

        poles = np.linalg.eigvals(A)
        all_poles.append(poles)

        # Plot in complex plane
        for pole in poles:
            axes[0].plot(pole.real, pole.imag, 'o', alpha=0.7,
                        markersize=8, label=f'{param_name}={val}')

    axes[0].axvline(x=0, color='k', linewidth=1)
    axes[0].axhline(y=0, color='k', linewidth=1)
    axes[0].set_xlabel('Real Part')
    axes[0].set_ylabel('Imaginary Part')
    axes[0].set_title('Pole Locations')

    # Plot unstable pole magnitude vs parameter
    unstable_poles = [max(p.real for p in poles) for poles in all_poles]
    axes[1].plot(values, unstable_poles, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Largest Real Part')
    axes[1].set_title('Unstable Pole vs Parameter')
    axes[1].axhline(y=0, color='k', linestyle='--')

    plt.suptitle(f'Effect of {param_name} on System Poles')
    plt.tight_layout()
    plt.show()

# Study pole length
parameter_study('L', [0.2, 0.3, 0.5, 0.7, 1.0])

# Study cart mass
parameter_study('m_c', [0.5, 1.0, 2.0, 5.0, 10.0])''')

        self._add_footer(nb,
            checkpoints=[
                "Can determine stability from eigenvalues",
                "Understand the relationship between pole locations and dynamics",
                "Can check controllability using the controllability matrix",
                "Explored how parameters affect stability"
            ],
            pitfalls=[
                "Confusing stability of the system with stability of the equilibrium",
                "Forgetting that complex eigenvalues come in conjugate pairs",
                "Not checking controllability before designing a controller"
            ],
            next_topic="P Control"
        )

        filepath = os.path.join(self.output_dir, "04_stability_analysis.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_p_control_notebook(self) -> str:
        """Create P Control notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Proportional (P) Control",
            objectives=[
                "Understand proportional control and gain selection",
                "Implement a P controller for the inverted pendulum",
                "Analyze the effect of proportional gain on stability",
                "Recognize the limitations of P-only control"
            ],
            prereqs=["Stability Analysis"]
        )

        self._add_setup_cell(nb)
        self._add_pendulum_class(nb)
        self._add_simulation_helper(nb)

        nb.add_markdown(r"""## Proportional Control

The simplest form of feedback control:

$$u = -K_p \cdot e(t)$$

where:
- $u$ = control input (force)
- $K_p$ = proportional gain
- $e(t) = \theta - \theta_{ref}$ = angle error

### How It Works
- **Error is positive** (pole tilting right) → Apply negative force (push left)
- **Error is negative** (pole tilting left) → Apply positive force (push right)
- **Larger error** → **Stronger correction**

### Closed-Loop System

With P control, the closed-loop dynamics become:
$$\dot{\mathbf{x}} = (A - BK_p C) \mathbf{x}$$

where $C$ selects the angle from the state vector.
""")

        nb.add_code('''def make_p_controller(Kp, theta_ref=np.pi):
    """Create a proportional controller."""
    def controller(state, t):
        x, x_dot, theta, theta_dot = state
        error = wrap_angle(theta - theta_ref)
        force = -Kp * error
        return force
    return controller

# Test different proportional gains
pendulum = InvertedPendulum(dt=0.02)
initial_state = [0, 0, np.pi + 0.1, 0]  # 5.7 degrees from upright

gains = [10, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for Kp, ax in zip(gains, axes.flat):
    controller = make_p_controller(Kp)
    result = simulate(pendulum, controller, 500, initial_state)

    t = result['time']
    theta = result['states'][:, 2]

    ax.plot(t, np.degrees(theta), 'b-', linewidth=2)
    ax.axhline(y=180, color='g', linestyle='--', label='Reference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(f'Kp = {Kp}')
    ax.set_ylim([150, 210])
    ax.legend()

plt.suptitle('Effect of Proportional Gain')
plt.tight_layout()
plt.show()

print("Observation: Higher Kp gives faster response but more oscillation!")''')

        nb.add_markdown("""## Analyzing Closed-Loop Poles

P control changes the pole locations. Let's see how!
""")

        nb.add_code('''def analyze_p_control_poles(Kp_values):
    """Analyze how P control changes the closed-loop poles."""
    # System matrices (linearized at upright)
    m_c, m_p, L, g = 1.0, 0.1, 0.5, 9.8
    denom = m_c + m_p

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, (m_p * g) / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (denom * g) / (L * m_c), 0]
    ])

    B = np.array([[0], [1/denom], [0], [1/(L * m_c)]])

    # P control feedback: u = -Kp * theta
    # This is equivalent to u = -Kp * [0 0 1 0] * x
    C = np.array([[0, 0, 1, 0]])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(Kp_values)))

    for Kp, color in zip(Kp_values, colors):
        K = Kp * C
        A_cl = A - B @ K
        poles = np.linalg.eigvals(A_cl)

        for pole in poles:
            ax.plot(pole.real, pole.imag, 'o', color=color,
                   markersize=10, alpha=0.7)

        # Connect with line for visibility
        ax.plot([p.real for p in poles], [p.imag for p in poles],
               '--', color=color, alpha=0.3)

    # Add stability boundary
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvspan(-15, 0, alpha=0.1, color='green')
    ax.axvspan(0, 5, alpha=0.1, color='red')

    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Closed-Loop Poles vs Proportional Gain', fontsize=14)
    ax.set_xlim([-15, 5])
    ax.set_ylim([-15, 15])

    # Add colorbar for Kp
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(Kp_values),
                                                  vmax=max(Kp_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Kp', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.show()

# Analyze for range of gains
Kp_values = np.linspace(0, 150, 20)
analyze_p_control_poles(Kp_values)

print("Notice: As Kp increases, poles move but may not all become stable!")
print("P-control alone may NOT be sufficient to stabilize the system.")''')

        nb.add_markdown("""## Limitations of P Control

P control has fundamental limitations:

1. **No damping**: Results in oscillatory behavior
2. **Doesn't consider velocity**: Can't predict where the system is heading
3. **May not stabilize all poles**: Some systems need more sophisticated control

### Why Does It Oscillate?

The proportional controller only reacts to the current error, not how fast the error is changing. When the pole crosses the setpoint, the error changes sign but the pole is still moving!
""")

        nb.add_code('''# Demonstrate oscillation problem

pendulum = InvertedPendulum(dt=0.02)
controller = make_p_controller(Kp=50)
result = simulate(pendulum, controller, 800, [0, 0, np.pi + 0.2, 0])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

t = result['time']
theta = np.degrees(result['states'][:, 2])
theta_dot = np.degrees(result['states'][:, 3])

# Phase portrait
axes[0].plot(theta, theta_dot, 'b-', linewidth=1)
axes[0].plot(theta[0], theta_dot[0], 'go', markersize=10, label='Start')
axes[0].plot(theta[-1], theta_dot[-1], 'ro', markersize=10, label='End')
axes[0].axvline(x=180, color='g', linestyle='--', alpha=0.5)
axes[0].axhline(y=0, color='g', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Angle (deg)')
axes[0].set_ylabel('Angular Velocity (deg/s)')
axes[0].set_title('Phase Portrait: The Spiral Shows Oscillation')
axes[0].legend()

# Time response
axes[1].plot(t, theta - 180, 'b-', linewidth=2, label='Angle Error')
axes[1].axhline(y=0, color='g', linestyle='--')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Angle Error (deg)')
axes[1].set_title('Error vs Time: Oscillations Persist')
axes[1].legend()

plt.tight_layout()
plt.show()

print("The phase portrait shows a spiral - oscillation with slow decay.")
print("To fix this, we need to add DAMPING with a derivative term!")''')

        nb.add_markdown("""## Task: Optimal P Gain? (15 mins)

Is there an "optimal" proportional gain?

1. Try to find a Kp that minimizes oscillation
2. Measure the settling time (time to reach within 2% of setpoint)
3. Measure the overshoot (maximum deviation past setpoint)

**Question**: Can you achieve zero oscillation with P control alone?
""")

        nb.add_code('''# TODO: Find the best P gain

def evaluate_controller(pendulum, controller, initial_state, steps=500):
    """Evaluate controller performance."""
    result = simulate(pendulum, controller, steps, initial_state)

    theta = result['states'][:, 2]
    error = np.abs(wrap_angle(theta - np.pi))

    # Metrics
    final_error = error[-1]
    max_error = np.max(error)
    settling_idx = np.where(error < 0.02)[0]  # 2% threshold (~1 degree)
    settling_time = settling_idx[0] * pendulum.dt if len(settling_idx) > 0 else np.inf

    return {
        'final_error': np.degrees(final_error),
        'max_error': np.degrees(max_error),
        'settling_time': settling_time
    }

# Your code here: test different Kp values and find the best one
Kp_test = [20, 40, 60, 80, 100, 120]
pendulum = InvertedPendulum(dt=0.02)

print("Kp     Final Error  Max Error  Settling Time")
print("-" * 50)
for Kp in Kp_test:
    controller = make_p_controller(Kp)
    metrics = evaluate_controller(pendulum, controller, [0, 0, np.pi + 0.1, 0])
    print(f"{Kp:3d}    {metrics['final_error']:6.3f}°      {metrics['max_error']:6.2f}°     {metrics['settling_time']:.2f}s")''')

        self._add_footer(nb,
            checkpoints=[
                "Implemented a proportional controller",
                "Understand how Kp affects closed-loop poles",
                "Observed the oscillation problem with P-only control",
                "Analyzed settling time and overshoot metrics"
            ],
            pitfalls=[
                "Setting Kp too high causes violent oscillations",
                "Forgetting to wrap the angle error to [-pi, pi]",
                "Expecting P control alone to give perfect tracking"
            ],
            next_topic="PD Control"
        )

        filepath = os.path.join(self.output_dir, "05_p_control.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_pd_control_notebook(self) -> str:
        """Create PD Control notebook."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="Proportional-Derivative (PD) Control",
            objectives=[
                "Understand how derivative control adds damping",
                "Implement a PD controller for the inverted pendulum",
                "Tune Kp and Kd for optimal performance",
                "Analyze the effect of derivative gain on stability"
            ],
            prereqs=["P Control"]
        )

        self._add_setup_cell(nb)
        self._add_pendulum_class(nb)
        self._add_simulation_helper(nb)

        nb.add_markdown(r"""## Adding Derivative Control

The derivative term predicts where the system is heading:

$$u = -K_p \cdot e(t) - K_d \cdot \dot{e}(t)$$

For our problem, $\dot{e} = \dot{\theta}$ (angular velocity), so:

$$u = -K_p \cdot (\theta - \theta_{ref}) - K_d \cdot \dot{\theta}$$

### How It Works
- When the pole is **moving toward** the setpoint → reduce the correction
- When the pole is **moving away** → increase the correction
- This **anticipates** the future error and provides **damping**

### Intuition
Think of a door closer:
- The spring (P) pulls the door shut
- The damper (D) prevents it from slamming
""")

        nb.add_code('''def make_pd_controller(Kp, Kd, theta_ref=np.pi):
    """Create a PD controller."""
    def controller(state, t):
        x, x_dot, theta, theta_dot = state
        error = wrap_angle(theta - theta_ref)
        force = -Kp * error - Kd * theta_dot
        return force
    return controller

# Compare P vs PD control
pendulum = InvertedPendulum(dt=0.02)
initial_state = [0, 0, np.pi + 0.15, 0]

# P controller
controller_p = make_p_controller(Kp=50)
result_p = simulate(pendulum, controller_p, 500, initial_state)

# PD controller (same Kp, add Kd)
controller_pd = make_pd_controller(Kp=50, Kd=10)
result_pd = simulate(pendulum, controller_pd, 500, initial_state)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

t = result_p['time']

# Angle comparison
axes[0, 0].plot(t, np.degrees(result_p['states'][:, 2]), 'b-',
               label='P only', linewidth=2)
axes[0, 0].plot(t, np.degrees(result_pd['states'][:, 2]), 'r-',
               label='PD', linewidth=2)
axes[0, 0].axhline(y=180, color='g', linestyle='--')
axes[0, 0].set_ylabel('Angle (deg)')
axes[0, 0].set_title('Pole Angle')
axes[0, 0].legend()

# Error comparison
axes[0, 1].plot(t, np.degrees(result_p['states'][:, 2]) - 180, 'b-', linewidth=2)
axes[0, 1].plot(t, np.degrees(result_pd['states'][:, 2]) - 180, 'r-', linewidth=2)
axes[0, 1].axhline(y=0, color='g', linestyle='--')
axes[0, 1].set_ylabel('Error (deg)')
axes[0, 1].set_title('Angle Error')

# Angular velocity
axes[1, 0].plot(t, np.degrees(result_p['states'][:, 3]), 'b-', linewidth=2)
axes[1, 0].plot(t, np.degrees(result_pd['states'][:, 3]), 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
axes[1, 0].set_title('Angular Velocity')

# Control force
axes[1, 1].plot(t, result_p['forces'], 'b-', linewidth=2)
axes[1, 1].plot(t, result_pd['forces'], 'r-', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Force (N)')
axes[1, 1].set_title('Control Force')

plt.suptitle('P Control vs PD Control', fontsize=14)
plt.tight_layout()
plt.show()

print("The derivative term dramatically reduces oscillation!")''')

        nb.add_markdown("""## Effect of Kd on Closed-Loop Poles

The derivative term adds damping to the closed-loop poles:
""")

        nb.add_code('''def analyze_pd_poles(Kp, Kd_values):
    """Show how Kd affects closed-loop poles."""
    m_c, m_p, L, g = 1.0, 0.1, 0.5, 9.8
    denom = m_c + m_p

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, (m_p * g) / denom, 0],
        [0, 0, 0, 1],
        [0, 0, (denom * g) / (L * m_c), 0]
    ])

    B = np.array([[0], [1/denom], [0], [1/(L * m_c)]])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(Kd_values)))

    for Kd, color in zip(Kd_values, colors):
        # PD control: u = -Kp*theta - Kd*theta_dot
        # K = [0, 0, Kp, Kd]
        K = np.array([[0, 0, Kp, Kd]])
        A_cl = A - B @ K
        poles = np.linalg.eigvals(A_cl)

        for pole in poles:
            ax.plot(pole.real, pole.imag, 'o', color=color, markersize=10)

    ax.axvline(x=0, color='k', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvspan(-20, 0, alpha=0.1, color='green')
    ax.axvspan(0, 5, alpha=0.1, color='red')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(f'Closed-Loop Poles: Kp={Kp}, varying Kd')
    ax.set_xlim([-20, 5])
    ax.set_ylim([-15, 15])

    sm = plt.cm.ScalarMappable(cmap='plasma',
                               norm=plt.Normalize(vmin=min(Kd_values),
                                                  vmax=max(Kd_values)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Kd')

    plt.grid(True, alpha=0.3)
    plt.show()

# Show effect of Kd
Kd_values = np.linspace(0, 15, 16)
analyze_pd_poles(Kp=50, Kd_values=Kd_values)

print("Notice: As Kd increases, oscillatory poles become more damped!")
print("(They move left and the imaginary part decreases)")''')

        nb.add_markdown("""## Tuning Kp and Kd

Finding the right balance between Kp (speed) and Kd (damping):
""")

        nb.add_code('''def tune_pd(Kp_range, Kd_range, initial_state=[0, 0, np.pi + 0.1, 0]):
    """Grid search for best Kp, Kd combination."""
    pendulum = InvertedPendulum(dt=0.02)

    results = []
    for Kp in Kp_range:
        for Kd in Kd_range:
            controller = make_pd_controller(Kp, Kd)
            sim_result = simulate(pendulum, controller, 500, initial_state)

            theta = sim_result['states'][:, 2]
            error = wrap_angle(theta - np.pi)

            # Performance metrics
            rms_error = np.sqrt(np.mean(error**2))
            max_force = np.max(np.abs(sim_result['forces']))
            final_error = np.abs(error[-1])

            # Combined cost (lower is better)
            cost = rms_error + 0.01 * max_force

            results.append({
                'Kp': Kp, 'Kd': Kd,
                'rms_error': np.degrees(rms_error),
                'max_force': max_force,
                'cost': cost
            })

    # Find best
    best = min(results, key=lambda x: x['cost'])
    return results, best

# Grid search
Kp_range = [30, 40, 50, 60, 70, 80]
Kd_range = [2, 4, 6, 8, 10, 12, 14]

results, best = tune_pd(Kp_range, Kd_range)

print(f"Best parameters: Kp={best['Kp']}, Kd={best['Kd']}")
print(f"RMS Error: {best['rms_error']:.3f}°, Max Force: {best['max_force']:.2f}N")

# Visualize as heatmap
costs = np.array([r['cost'] for r in results]).reshape(len(Kp_range), len(Kd_range))

plt.figure(figsize=(10, 6))
plt.imshow(costs, aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Cost (lower is better)')
plt.xticks(range(len(Kd_range)), Kd_range)
plt.yticks(range(len(Kp_range)), Kp_range)
plt.xlabel('Kd')
plt.ylabel('Kp')
plt.title('PD Tuning: Cost Function')

# Mark best
best_idx = np.argmin(costs)
best_i, best_j = np.unravel_index(best_idx, costs.shape)
plt.plot(best_j, best_i, 'r*', markersize=20)

plt.show()''')

        nb.add_markdown("""## Task: Optimal PD Tuning (20 mins)

1. Use the tuning grid to find good Kp, Kd values
2. Test the tuned controller with larger initial disturbances
3. What happens when Kd is too high? (Hint: noise sensitivity)

**Challenge**: Can PD control handle a disturbance force? Try adding a constant force during simulation.
""")

        nb.add_code('''# TODO: Test your tuned PD controller

# Your best gains from tuning
Kp_best = 60  # Modify based on your results
Kd_best = 10

controller = make_pd_controller(Kp_best, Kd_best)
pendulum = InvertedPendulum(dt=0.02)

# Test with larger disturbance
large_disturbance = [0, 0, np.pi + 0.3, 0]  # ~17 degrees
result = simulate(pendulum, controller, 500, large_disturbance)
plot_response(result, f"PD Control: Kp={Kp_best}, Kd={Kd_best}")

# Test with external force disturbance
def disturbance_force(t):
    return 5.0 if 2.0 < t < 2.5 else 0.0

result_dist = simulate(pendulum, controller, 500, [0, 0, np.pi, 0],
                       disturbance=disturbance_force)
plot_response(result_dist, "PD Control with Force Disturbance")''')

        self._add_footer(nb,
            checkpoints=[
                "Implemented PD controller and understand the role of Kd",
                "Observed how derivative control reduces oscillation",
                "Successfully tuned Kp and Kd for good performance",
                "Tested controller with disturbances"
            ],
            pitfalls=[
                "Setting Kd too high amplifies measurement noise",
                "Forgetting that Kd only helps with oscillation, not steady-state error",
                "Not considering the trade-off between response speed and damping"
            ],
            next_topic="PID Control"
        )

        filepath = os.path.join(self.output_dir, "06_pd_control.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_pid_control_notebook(self) -> str:
        """Create PID Control notebook - placeholder for now."""
        nb = NotebookBuilder()

        self._add_header(nb,
            title="PID Control and Tuning",
            objectives=[
                "Understand the role of integral control",
                "Implement a complete PID controller",
                "Apply systematic tuning methods (Ziegler-Nichols)",
                "Handle integral windup"
            ],
            prereqs=["PD Control"]
        )

        self._add_setup_cell(nb)
        self._add_pendulum_class(nb)
        self._add_simulation_helper(nb)

        nb.add_markdown(r"""## The Complete PID Controller

PID adds an **integral term** to eliminate steady-state error:

$$u = -K_p \cdot e(t) - K_i \int_0^t e(\tau) d\tau - K_d \cdot \dot{e}(t)$$

### The Three Terms:
- **P (Proportional)**: Responds to current error
- **I (Integral)**: Accumulates past errors → eliminates steady-state error
- **D (Derivative)**: Predicts future error → adds damping
""")

        nb.add_code('''def make_pid_controller(Kp, Ki, Kd, theta_ref=np.pi, dt=0.02,
                           anti_windup=True, windup_limit=10.0):
    """Create a PID controller with anti-windup."""
    integral = [0.0]  # Use list to allow modification in closure

    def controller(state, t):
        x, x_dot, theta, theta_dot = state
        error = wrap_angle(theta - theta_ref)

        # Update integral
        integral[0] += error * dt

        # Anti-windup: limit integral term
        if anti_windup:
            integral[0] = np.clip(integral[0], -windup_limit, windup_limit)

        # PID control law
        force = -Kp * error - Ki * integral[0] - Kd * theta_dot
        return force

    return controller

# Compare P, PD, and PID
pendulum = InvertedPendulum(dt=0.02)
initial_state = [0, 0, np.pi + 0.1, 0]

Kp, Ki, Kd = 50, 1.0, 8

controller_p = make_p_controller(Kp)
controller_pd = make_pd_controller(Kp, Kd)
controller_pid = make_pid_controller(Kp, Ki, Kd)

result_p = simulate(pendulum, controller_p, 600, initial_state)
result_pd = simulate(pendulum, controller_pd, 600, initial_state)
result_pid = simulate(pendulum, controller_pid, 600, initial_state)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
t = result_p['time']

for result, label, color in [(result_p, 'P', 'blue'),
                              (result_pd, 'PD', 'orange'),
                              (result_pid, 'PID', 'green')]:
    axes[0, 0].plot(t, np.degrees(result['states'][:, 2]) - 180,
                   color=color, label=label, linewidth=2)
    axes[0, 1].plot(t, result['states'][:, 0], color=color, linewidth=2)
    axes[1, 0].plot(t, result['forces'], color=color, linewidth=2)

axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[0, 0].set_ylabel('Angle Error (deg)')
axes[0, 0].set_title('Angle Error')
axes[0, 0].legend()

axes[0, 1].set_ylabel('Cart Position (m)')
axes[0, 1].set_title('Cart Position')

axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Force (N)')
axes[1, 0].set_title('Control Force')

axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.5, f'Kp={Kp}\\nKi={Ki}\\nKd={Kd}',
               fontsize=16, ha='center', va='center',
               transform=axes[1, 1].transAxes)

plt.suptitle('Comparison: P vs PD vs PID Control')
plt.tight_layout()
plt.show()''')

        nb.add_markdown("""## Integral Windup

A common problem with integral control: if the error persists (e.g., due to saturation), the integral term grows unboundedly.

**Solutions:**
1. **Clamping**: Limit the integral term
2. **Back-calculation**: Reduce integral when output saturates
3. **Conditional integration**: Only integrate when near setpoint
""")

        nb.add_code('''# Demonstrate integral windup

def make_pid_with_saturation(Kp, Ki, Kd, force_limit=20.0, anti_windup=False):
    """PID with force saturation to show windup effects."""
    integral = [0.0]
    dt = 0.02

    def controller(state, t):
        x, x_dot, theta, theta_dot = state
        error = wrap_angle(theta - np.pi)
        integral[0] += error * dt

        if anti_windup:
            integral[0] = np.clip(integral[0], -5, 5)

        force = -Kp * error - Ki * integral[0] - Kd * theta_dot
        force_sat = np.clip(force, -force_limit, force_limit)
        return force_sat

    return controller

# Compare with and without anti-windup
pendulum = InvertedPendulum(dt=0.02)
initial_state = [0, 0, np.pi + 0.3, 0]  # Larger disturbance

ctrl_no_aw = make_pid_with_saturation(50, 2.0, 8, anti_windup=False)
ctrl_with_aw = make_pid_with_saturation(50, 2.0, 8, anti_windup=True)

result_no_aw = simulate(pendulum, ctrl_no_aw, 800, initial_state)
result_with_aw = simulate(pendulum, ctrl_with_aw, 800, initial_state)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t = result_no_aw['time']
axes[0].plot(t, np.degrees(result_no_aw['states'][:, 2]) - 180,
            'r-', label='No Anti-Windup', linewidth=2)
axes[0].plot(t, np.degrees(result_with_aw['states'][:, 2]) - 180,
            'g-', label='With Anti-Windup', linewidth=2)
axes[0].axhline(y=0, color='k', linestyle='--')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Angle Error (deg)')
axes[0].set_title('Effect of Anti-Windup')
axes[0].legend()

axes[1].plot(t, result_no_aw['forces'], 'r-', label='No Anti-Windup')
axes[1].plot(t, result_with_aw['forces'], 'g-', label='With Anti-Windup')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Force (N)')
axes[1].set_title('Control Force')
axes[1].legend()

plt.tight_layout()
plt.show()''')

        self._add_footer(nb,
            checkpoints=[
                "Implemented full PID controller",
                "Understand the role of integral action",
                "Know how to handle integral windup",
                "Can tune PID gains for good performance"
            ],
            pitfalls=[
                "Setting Ki too high causes overshoot and oscillation",
                "Forgetting anti-windup can cause instability",
                "Integral term can fight with position control objectives"
            ],
            next_topic="State Feedback Control"
        )

        filepath = os.path.join(self.output_dir, "07_pid_control.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    # Placeholder implementations for remaining notebooks
    def create_state_feedback_notebook(self) -> str:
        """Create State Feedback notebook."""
        nb = NotebookBuilder()
        self._add_header(nb, "Full State Feedback Control",
            ["Understand state feedback vs output feedback",
             "Design a full state feedback controller",
             "Implement simultaneous pole and cart control",
             "Analyze trade-offs in multi-objective control"],
            ["PID Control"])
        self._add_setup_cell(nb)
        nb.add_markdown("## Coming Soon\nThis notebook will cover full state feedback control.")
        filepath = os.path.join(self.output_dir, "08_state_feedback.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_pole_placement_notebook(self) -> str:
        """Create Pole Placement notebook."""
        nb = NotebookBuilder()
        self._add_header(nb, "Pole Placement Design",
            ["Understand the pole placement design method",
             "Use Ackermann's formula for gain calculation",
             "Design controllers for desired transient response",
             "Verify closed-loop pole locations"],
            ["State Feedback"])
        self._add_setup_cell(nb)
        nb.add_markdown("## Coming Soon\nThis notebook will cover pole placement design.")
        filepath = os.path.join(self.output_dir, "09_pole_placement.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_disturbance_rejection_notebook(self) -> str:
        """Create Disturbance Rejection notebook."""
        nb = NotebookBuilder()
        self._add_header(nb, "Disturbance Rejection and Robustness",
            ["Understand disturbance rejection requirements",
             "Test controller robustness to parameter variations",
             "Implement integral action for disturbance rejection",
             "Analyze sensitivity functions"],
            ["PID Control"])
        self._add_setup_cell(nb)
        nb.add_markdown("## Coming Soon\nThis notebook will cover disturbance rejection.")
        filepath = os.path.join(self.output_dir, "10_disturbance_rejection.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_observer_design_notebook(self) -> str:
        """Create Observer Design notebook."""
        nb = NotebookBuilder()
        self._add_header(nb, "State Observers and Estimation",
            ["Understand the need for state estimation",
             "Design a Luenberger observer",
             "Implement observer-based control",
             "Analyze separation principle"],
            ["State Feedback"])
        self._add_setup_cell(nb)
        nb.add_markdown("## Coming Soon\nThis notebook will cover observer design.")
        filepath = os.path.join(self.output_dir, "11_observer_design.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def create_lqr_control_notebook(self) -> str:
        """Create LQR Control notebook."""
        nb = NotebookBuilder()
        self._add_header(nb, "Linear Quadratic Regulator (LQR)",
            ["Understand optimal control concepts",
             "Formulate the LQR problem",
             "Solve the Riccati equation",
             "Tune Q and R matrices for desired behavior"],
            ["State Feedback", "Pole Placement"])
        self._add_setup_cell(nb)
        nb.add_markdown("## Coming Soon\nThis notebook will cover LQR control design.")
        filepath = os.path.join(self.output_dir, "12_lqr_control.ipynb")
        nb.save(filepath)
        self.log(f"Created: {filepath}")
        return filepath

    def interactive_menu(self):
        """Interactive menu for generating notebooks."""
        while True:
            print("\n" + "="*60)
            print("INSTRUCTOR AGENT - Notebook Generator")
            print("="*60)
            self.list_topics()
            print("\nOptions:")
            print("  Enter topic name to generate that notebook")
            print("  'all' - Generate all notebooks")
            print("  'list' - Show available topics")
            print("  'quit' - Exit")
            print("-"*60)

            try:
                choice = input("Enter choice: ").strip().lower()
            except EOFError:
                break

            if choice == 'quit' or choice == 'q':
                print("Goodbye!")
                break
            elif choice == 'all':
                created = self.create_all_notebooks()
                print(f"\nCreated {len(created)} notebooks in {self.output_dir}/")
            elif choice == 'list':
                self.list_topics()
            elif choice in self.topics:
                filepath = self.create_notebook(choice)
                if filepath:
                    print(f"\nCreated: {filepath}")
            else:
                # Try to match by number
                try:
                    idx = int(choice) - 1
                    topic = list(self.topics.keys())[idx]
                    filepath = self.create_notebook(topic)
                    if filepath:
                        print(f"\nCreated: {filepath}")
                except (ValueError, IndexError):
                    print(f"Unknown topic: {choice}")


def main():
    parser = argparse.ArgumentParser(
        description="Instructor Agent - Generate educational notebooks for control systems"
    )
    parser.add_argument("--topic", "-t", type=str,
                       help="Generate a specific topic notebook")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available topics")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Generate all notebooks")
    parser.add_argument("--output", "-o", type=str, default="generated_notebooks",
                       help="Output directory for notebooks")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")

    args = parser.parse_args()

    agent = InstructorAgent(output_dir=args.output, verbose=not args.quiet)

    if args.list:
        agent.list_topics()
    elif args.all:
        created = agent.create_all_notebooks()
        print(f"\nCreated {len(created)} notebooks in {args.output}/")
    elif args.topic:
        filepath = agent.create_notebook(args.topic)
        if not filepath:
            print(f"Unknown topic: {args.topic}")
            agent.list_topics()
    else:
        agent.interactive_menu()


if __name__ == "__main__":
    main()
