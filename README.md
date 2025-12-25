# ESP2110-inverted-pendulum-lab

## Lesson Plan (Notebook Sequence)
This repository is a set of ESP2110 inverted pendulum lab notebooks. Follow the notebooks in order; each file in `notebook_for_reference/` is a self-contained lesson.

### Lesson 1: Setting Up
- Notebook: `notebook_for_reference/ESP2110 - Lab Lesson 1 (Setting Up).ipynb`
- Focus: Task description, environment setup, and system dynamics overview.
- Task: Understand the system dynamics (15 mins).

### Lesson 2: Modeling
- Notebook: `notebook_for_reference/ESP2110 - Lab Lesson 2 (Modeling).ipynb`
- Focus: Modeling and linearization.
- Task: Linearize the system at specified positions (30 mins).

### Lesson 3: State Space
- Notebook: `notebook_for_reference/ESP2110 - Lab Lesson 3 (State Space).ipynb`
- Focus: Linearization at upward and downward equilibria; state-space form.
- Task: Determine stability of the upward and downward systems when `f = 0` (15 mins).

### Lesson 4: Controllers and Disturbances
- Implementing Controller: `notebook_for_reference/ESP2110 - Lab Lesson 4 (Implementing Controller).ipynb`
  - Topics: Eigenvalue analysis and PID control.
  - Task: Implement P, PD, and PID controllers (through end of lesson).
  - Answer key: `notebook_for_reference/ESP2110 - Lab Lesson 4 (Implementing Controller) (Answer).ipynb`
- Cart Controller: `notebook_for_reference/ESP2110 - Lab Lesson 4 (Cart Controller).ipynb`
  - Tasks: Cascaded control for pole + cart stabilization; setpoint changes.
- External Disturbances: `notebook_for_reference/ESP2110 - Lab Lesson 4 (External Disturbances).ipynb`
  - Task: Investigate the effect of disturbances on the system.

### Lesson 5: State Estimation & Robust Control
- Notebook: `notebook_for_reference/ESP2110 - Lab Lesson 5 (State Estimation & Robust Control).ipynb`
- Focus: Noisy measurements, state estimation, and robustness to model mismatch.
- Tasks:
  - Add measurement noise to simulated sensors.
  - Implement a state estimator (Kalman or Luenberger).
  - Re-tune the controller using estimated states.
  - Stress test with parameter drift (mass/length) and check stability margins.

## Running the Lessons
- Start JupyterLab: `jupyter lab`
- Or use the classic UI: `jupyter notebook`
- Open notebooks in order and run cells top to bottom for each lesson.
