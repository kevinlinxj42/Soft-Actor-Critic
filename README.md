# Soft Actor-Critic (PyTorch + Gymnasium)

A reproducible Soft Actor-Critic training pipeline targeting MuJoCo v4 continuous control tasks.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
export PYTHONPATH=src
python scripts/train.py --config configs/sac_mujoco_base.yaml
```

## Layout

- `src/sac/`: SAC implementation modules
- `scripts/train.py`: training entrypoint
- `scripts/eval.py`: evaluation entrypoint
- `scripts/run_sweep.py`: ablation sweep launcher
- `scripts/aggregate_results.py`: aggregate multi-seed metrics
- `configs/`: baseline and sweep configs
- `tests/`: unit, integration, and regression tests
