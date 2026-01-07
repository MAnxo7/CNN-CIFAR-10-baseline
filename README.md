# CIFAR-10 CNN — Experiment-driven baseline

A small PyTorch project where I trained a CNN on **CIFAR-10** and iteratively improved it through a structured experiment log (architecture, regularization, data augmentation, schedulers, and training length).

**Main goals:**
- Build a clean training loop with checkpointing + metrics export
- Keep an experiment log and compare changes **with evidence** (loss/acc curves + CSV)
- Add a simple CLI to train / evaluate / run inference on a single image

## Results (best runs)

- Best eval accuracy (my runs): **~0.865** (see `EXPERIMENTO 20/21`) (**“Best checkpoint: runs/EXPERIMENTO 21/best.pt”**)
- Main finding: **increasing feature depth helped more than increasing channel capacity** (for my setup)

For the full experiment history and decisions, see: **`experiments_log.md`**.

---

## Project structure

Typical layout:

```text
.
├── main.py
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── utils.py
│   └── viz.py
├── runs/
│   ├── {run_date}/
│   │   ├── figures/
│   │   │   ├── acc.jpg
│   │   │   └── loss.jpg
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── metrics.csv
│   └── ...
└── experiments_log.md
````

Each `runs/{run_date}/` folder stores:

* `metrics.csv` with per-epoch metrics
* `loss.png`, `acc.png` curves
* `best.pt` (best validation checkpoint) and `last.pt` (latest checkpoint)

---

## Setup

### 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision matplotlib
```

> If you want exact reproducibility, pin versions (`pip freeze > requirements.txt`).

### 2) Dataset

CIFAR-10 downloads automatically on first run (TorchVision).

---

## Usage

### Train (default architecture)

```bash
python3 main.py --epochs 50 --batch-size 128 --lr 0.001
```

### Select architecture

This repo includes the final/best architectures as selectable variants:

* `exp0` (initial baseline)
* `exp15` (strong baseline)
* `exp21` (deeper features; best overall in my runs)

```bash
python3 main.py --arch exp21 --epochs 70 --lr 0.001
```

### Disable scheduler

By default, the best setup uses:

```python
MultiStepLR(milestones=[30, 40], gamma=0.3)
```

Disable it with:

```bash
python3 main.py --disable-sched
```

### Continue training from a checkpoint

```bash
python3 main.py --ckpt-path "runs/EXPERIMENTO 21/last.pt" --epochs 70
```

The script loads the checkpoint and trains only the remaining epochs.

### Evaluate from a checkpoint

```bash
python3 main.py --eval-only --ckpt-path "runs/EXPERIMENTO 21/best.pt"
```

### Inference on a single image

You can optionally pass an image path. The script resizes it to CIFAR-10 format and prints top-3 predictions.

Evaluate + inference (recommended):

```bash
python3 main.py \
  --eval-only \
  --ckpt-path "runs/EXPERIMENTO 21/best.pt" \
  --image-path "/path/to/image.jpg"
```

Train then inference:

```bash
python3 main.py --epochs 50 --image-path "/path/to/image.jpg"
```

**Note:** CIFAR-10 classes are limited to:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.
Random real-world images may not map well to these classes.

---

## Reproducibility

* I use a fixed seed (`utils.set_seed(..., deterministic=True)`).
* Results still vary slightly depending on hardware / CUDA / PyTorch versions.
* Every run exports CSV + plots + checkpoints under `runs/`.

---

## Key takeaways (from my experiment log)

High-level summary (details in `experiments_log.md`):

* **Lower LR mattered:** switching from `lr=0.01` to `lr=0.001` made training much more stable.
* **Augmentations helped:** `RandomCrop` + `HorizontalFlip` improved eval performance.
* **ColorJitter/Rotation** were too expensive for my setup and did not improve results.
* **Weight decay (AdamW):** small impact compared to architecture changes.
* **Scheduler:** `MultiStepLR` helped when applied later with a milder gamma.
* **Architecture:** after a point, **depth helped more than width** (channels).

---

## Notes / limitations

* The inference CLI is meant as a small “demo”; CIFAR-10 is very small and domain-limited.
* Checkpoints are architecture-specific. Loading a checkpoint with the wrong `--arch` will fail.
* Training speed depends heavily on CPU dataloading and augmentations (`num_workers`, etc.).
* Only EXPERIMENT 0 (BASE), EXPERIMENT 15 and EXPERIMENT 21 checkpoints are included.

---

## License

MIT

