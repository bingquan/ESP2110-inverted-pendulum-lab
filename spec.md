# Notebook Improvement Spec

## Goal
Create improved versions of the ESP2110 inverted pendulum lab notebooks without modifying the originals. The improved notebooks must run in Google Colab and provide clearer learning flow, checkpoints, and common pitfalls.

## Scope
- Source notebooks live in `notebook_for_reference/`.
- Improved notebooks live in `updated_notebooks/` with the same filenames.
- Do not edit any files under `notebook_for_reference/`.
- Maintain compatibility with Colab by including a setup cell for dependencies.

## Structure Changes (Applied to Every Notebook)
1. **Standardized header**
   - Title using the lesson name.
   - Short Colab usage note.
   - Learning objectives (2-4 bullets).
   - Parameter table with common symbols (cart mass, pendulum mass, length, gravity, sample time).
   - Divider and a `Lesson Content` section that precedes the original content.
2. **Colab setup cell**
   - A code cell at the top to install `numpy`, `scipy`, and `matplotlib`.
   - The cell should be safe to re-run.
3. **End-of-lesson guidance**
   - A `Checkpoints` section listing expected verification items.
   - A `Common Pitfalls` section listing 2-4 likely errors.

## Lesson-Specific Enhancements
- Each lesson includes objectives and checkpoint items that match its topic:
  - Lesson 1: environment setup and dynamics overview.
  - Lesson 2: nonlinear model and linearization checks.
  - Lesson 3: state-space derivation and stability analysis.
  - Lesson 4A: controller implementation and tuning.
  - Lesson 4B: cascaded control and setpoint changes.
  - Lesson 4C: disturbance effects and mitigation.
  - Lesson 4 Answer: comparison guidance with reference results.

## Colab Compatibility Requirements
- No local file path assumptions inside new content.
- All new setup instructions should work in a clean Colab runtime.
- Keep the original code cells intact so existing computations still run.

## Deliverables
- `updated_notebooks/` containing improved notebooks with the additions above.
- `spec.md` describing the improvement plan (this file).
