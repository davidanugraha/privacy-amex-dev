# privacy-amex-dev

Multi-party privacy scenarios: a generator (`src/scenario_gen`) and an experiment runner (`src/experiments`).

## Example: generate a scenario and run it

Generate one cell **and** run it end-to-end, recording results into the bundle:

```bash
apptainer exec --writable-tmpfs --env-file .env --pwd /app \
  --bind ./src:/app/src --bind ./scenarios:/app/scenarios \
  privacy-experiment.sif \
  python -m src.scenario_gen.pipeline \
    --domain consumer_credit --topology fan_in --data-form aggregate --seed 0 \
    --output-dir scenarios/generated/test1 --run --repeat 1 --judge-mode agent
```

Generate only: drop `--run`. Run an already-generated bundle:

```bash
python -m src.scenario_gen.run --bundle scenarios/generated/test1 --repeat 1 --judge-mode agent
```

Run an existing hand-written scenario directly through the runner:

```bash
python -m src.experiments.runner scenarios/consumer_credit/tasks/<task>/scenario.yaml --repeat 1 --judge-mode agent
```
