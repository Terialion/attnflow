# 🔍 AttnFlow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Real-time Transformer Attention Memory Flow Visualization**

AttnFlow is a lightweight Python toolkit for tracking and visualizing Transformer attention KV-cache memory growth during inference.
It uses PyTorch forward hooks, so you can inspect memory behavior without modifying model internals.

## Navigation

- [🔍 AttnFlow](#-attnflow)
  - [Navigation](#navigation)
  - [Who This Is For](#who-this-is-for)
  - [Why AttnFlow](#why-attnflow)
  - [Project Status](#project-status)
  - [Documentation](#documentation)
  - [Quick Start (30 seconds)](#quick-start-30-seconds)
    - [1) Install](#1-install)
    - [2) Minimal Example](#2-minimal-example)
    - [3) Run Demos](#3-run-demos)
  - [Verify Setup](#verify-setup)
  - [Reading Paths](#reading-paths)

## Who This Is For

- **ML/AI engineers**: inspect attention memory behavior during inference quickly
- **Infra researchers**: compare KV growth across sequence lengths and layers
- **Framework/tooling developers**: extend hook matching and visualization modules

## Why AttnFlow

- Zero-intrusive tracking via forward hooks
- Per-layer memory timeline with CLI and matplotlib outputs
- Realtime dashboard demo (timeline + global KV growth bar)
- Strictly tested codebase (`pytest` green in current workspace)

## Project Status

AttnFlow is currently **Alpha** (`v0.1.0`).
APIs are usable for research and internal tooling, and may still evolve before a stable release.

## Documentation

- English docs: [docs/en.md](docs/en.md)
- 中文文档: [docs/zh-CN.md](docs/zh-CN.md)
- Docs index: [docs/README.md](docs/README.md)

## Quick Start (30 seconds)

### 1) Install

```bash
git clone https://github.com/Terialion/attnflow.git
cd attnflow
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
pip install -e .
```

### 2) Minimal Example

```python
from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary
import torch

model = your_transformer_model()
model.eval()

with AttentionTracker(model) as tracker:
    with torch.no_grad():
        _ = model(input_ids)

    print_memory_summary(tracker.get_memory_stats())
```

### 3) Run Demos

```bash
python demo/simple_demo.py
python demo/realtime_demo.py
```

- `realtime_demo.py` (recommended): real-time timeline + global KV growth bar
- `simple_demo.py`: baseline tracking + static visualization

> Windows note: `simple_demo.py` uses `/tmp/...` save paths by default and may fail on Windows unless you change the output path.

## Verify Setup

```bash
python -m pytest -q
```

If tests pass, your environment is correctly set up for development and demos.

## Reading Paths

- **I want to run it now (new users)**
  - Run `python demo/realtime_demo.py`
  - Read [docs/en.md](docs/en.md#quick-start)
- **I want architecture and API details (developers)**
  - Read [docs/en.md](docs/en.md#architecture-overview)
  - Read [docs/en.md](docs/en.md#attention-name-compatibility)
- **I prefer Chinese docs**
  - Read [docs/zh-CN.md](docs/zh-CN.md)
- **I need the full docs index**
  - Read [docs/README.md](docs/README.md)

---

**Made with ❤️ by AttnFlow Team**
