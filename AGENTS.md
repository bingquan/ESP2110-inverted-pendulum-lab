# Repository Guidelines

## Project Structure & Module Organization
- `README.md` contains the repository title only.
- `notebook_for_reference/` holds the lab notebooks for the ESP2110 inverted pendulum exercises (Lesson 1–4, plus an answer key). Each notebook is a self-contained learning module; there is no separate source or test directory.

## Build, Test, and Development Commands
- `jupyter lab` starts JupyterLab for opening and running the notebooks in `notebook_for_reference/`.
- `jupyter notebook` is an alternative launcher if you prefer the classic UI.
- No build system or automated test runner is configured in this repository.

## Coding Style & Naming Conventions
- Follow Python/Jupyter conventions inside notebooks: 4-space indentation, descriptive variable names, and clear cell separation between setup, modeling, control design, and analysis.
- Keep notebook names consistent with the existing pattern (e.g., `ESP2110 - Lab Lesson 4 (Cart Controller).ipynb`).
- Prefer adding short Markdown cells to explain equations, assumptions, or parameter choices.

## Testing Guidelines
- There is no dedicated testing framework or coverage target.
- When modifying notebooks, validate results by re-running all cells in order (`Kernel -> Restart & Run All`) and checking plots/outputs for regressions.

## Commit & Pull Request Guidelines
- Git history contains only an initial commit; no formal convention is established.
- Use concise, imperative commit messages (e.g., `Add lesson 3 state-space notes`).
- For PRs, include a brief summary, list of notebooks touched, and screenshots of key plots if outputs changed.

## Configuration & Environment Tips
- Use a Python environment with Jupyter installed; common dependencies for control labs include `numpy`, `scipy`, and `matplotlib`.
- Keep large data or generated artifacts out of the repo unless explicitly required by a lab.
