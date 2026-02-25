# AttnFlow English Documentation

[Back to README](../README.md)

## ✨ What AttnFlow Solves

AttnFlow helps you answer one practical question quickly:

**How does attention KV-cache memory grow across layers during inference?**

It hooks Transformer forward passes, records per-layer snapshots, and renders memory behavior in both CLI tables and matplotlib views.

## Core Capabilities

- 🎯 Zero-intrusive tracking using PyTorch forward hooks
- 📊 KV-cache size estimation from live tensor shapes
- 📈 Per-layer timeline query and plotting
- ⚡ Realtime dashboard: timeline + global KV growth bar
- 🧱 Modular architecture for easy extension (hooks / core / viz)

---

<a id="quick-start"></a>
## 🚀 Quick Start

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

- `realtime_demo.py` (recommended): real-time animation (per-layer timeline + global KV growth bar)
- `simple_demo.py`: baseline tracking + static visualization

> Windows note: `simple_demo.py` uses `/tmp/...` save paths by default and may fail on Windows. Use `realtime_demo.py` first, or change the save path.

### 4) Verify Environment

```bash
python -m pytest -q
```

---

## 📋 Real Output Sample

From `python demo/realtime_demo.py`:

```
[2026-02-25 20:47:35 - attnflow.hooks.transformer_hooks - INFO] Successfully registered 2 hooks

======================================================================
                       ATTENTION MEMORY SUMMARY
======================================================================
Layer Name                    Peak Memory (MB)    Max Seq Len
----------------------------------------------------------------------
layers.0.self_attention       0.07                23
layers.1.self_attention       0.07                23
----------------------------------------------------------------------
TOTAL                         0.13
======================================================================

[2026-02-25 20:47:41 - attnflow.hooks.transformer_hooks - INFO] All hooks unregistered
```

You may also see third-party logs (for example `MMKV`, `libpng warning`) depending on your environment. They do not affect AttnFlow tracking results.

---

## 🧠 Usage Patterns

### Pattern A: Context Manager (recommended)

```python
with AttentionTracker(model) as tracker:
    with torch.no_grad():
        _ = model(input_ids)

    stats = tracker.get_memory_stats()
    summary = tracker.get_summary()
```

### Pattern B: Manual Lifecycle

```python
tracker = AttentionTracker(model, enable_logging=True)
tracker.register_hooks()
tracker.start_tracking()

with torch.no_grad():
    _ = model(input_ids)

tracker.stop_tracking()
print(tracker.get_summary())
tracker.unregister_hooks()
```

### Pattern C: Multi-run Reset

```python
tracker = AttentionTracker(model)
tracker.register_hooks()

tracker.start_tracking()
_ = model(batch_a)
tracker.stop_tracking()

tracker.reset_stats()

tracker.start_tracking()
_ = model(batch_b)
tracker.stop_tracking()
```

---

## 📊 Visualization Options

### CLI Output

```python
from attnflow.viz import print_memory_summary, print_memory_timeline, print_all_timelines

stats = tracker.get_memory_stats()
print_memory_summary(stats)
print_memory_timeline(stats, "layers.0.self_attention")
print_all_timelines(stats)
```

### Static matplotlib Plots

```python
from attnflow.viz import Visualizer

viz = Visualizer(style="default")
fig1 = viz.plot_memory_timeline(stats)
fig2 = viz.plot_peak_memory_comparison(stats)
fig1.savefig("timeline.png", dpi=300, bbox_inches="tight")
fig2.savefig("peak.png", dpi=300, bbox_inches="tight")
```

### Realtime Dashboard (Day2)

```python
from attnflow.viz import RealtimeMemoryDashboard

dashboard = RealtimeMemoryDashboard(refresh_interval_ms=120)
fig, anim = dashboard.create_animation(stats)
```

---

<a id="architecture-overview"></a>
## 🏗️ Architecture Overview

| Layer | Module | Responsibility |
|---|---|---|
| Control | `attnflow/core/tracker.py` | Hook lifecycle + tracking orchestration |
| Interception | `attnflow/hooks/transformer_hooks.py` | Attention module detection + snapshot recording |
| Data | `attnflow/core/memory_stats.py` | Snapshot storage, summary and timeline query |
| Visualization | `attnflow/viz/` | CLI, static plots, realtime dashboard |
| Utilities | `attnflow/utils/` | Constants and logging |

### One-pass Data Flow

1. Model forward runs.
2. Hook triggers on matched attention modules.
3. Output tensor shape is extracted.
4. K/V bytes are estimated.
5. Snapshot is recorded into `MemoryStats`.
6. Summary / timeline is rendered by CLI or matplotlib.

---

<a id="attention-name-compatibility"></a>
## 🔌 Attention Name Compatibility

Current detection is **name-based** on the final module token, with conservative matching to avoid false positives.

### Recognized by default

- `attention`
- `self_attention`
- `self_attn`
- `cross_attn`
- `attn`
- `c_attn`
- `multihead`
- `multiheadattention`
- `SelfAttention` (normalized to snake_case before matching)

### Not matched by default (intentional)

- `attention_mask`
- `attn_dropout`
- `q_proj`, `k_proj`, `v_proj`

---

## ❓ FAQ

### Why are some layers not tracked?

Most cases are naming-related. Print module names and check whether your attention block names match supported patterns.

```python
for name, _ in model.named_modules():
    if "attn" in name.lower() or "attention" in name.lower():
        print(name)
```

### Why does reported memory look approximate?

Current implementation estimates cache bytes from output shape and dtype assumptions. This is fast and model-agnostic. For custom attention kernels, consider extending hook logic to read actual cache tensors.

### Why is `max_seq_len=512` but report shows `Max Seq Len=32`?

`max_seq_len` is model capacity. Reported sequence length is the maximum value actually seen in your run.

---

## ✅ Best Practices

- Use `model.eval()` + `torch.no_grad()` during tracking.
- Prefer context-manager API to guarantee hook cleanup.
- Run multiple sequence lengths in one session to observe growth behavior.
- Keep demo and docs output synchronized with real runs.
- Add regression tests whenever changing hook matching logic.

---

## 🧪 Development Commands

```bash
python -m pytest -q
python -m pytest -q tests/test_hooks.py
python demo/realtime_demo.py
```

---

## Roadmap

- Better support for diverse attention implementations (FlashAttention variants)
- More configurable matching strategy (custom matcher injection)
- Interactive dashboard backend options (Plotly / Web)

---

## Related Docs

- Chinese full documentation: [zh-CN.md](zh-CN.md)
- Documentation index: [README.md](README.md)

---

**Made with ❤️ by AttnFlow Team**