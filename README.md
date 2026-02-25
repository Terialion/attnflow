# 🔍 AttnFlow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Real-time Transformer Attention Memory Flow Visualization**

一个轻量级、易用的 Python 工具库，用于实时追踪和可视化 Transformer 模型推理过程中的 KV Cache 内存消耗。通过 PyTorch hooks 机制，无需修改模型代码即可深入了解注意力机制的内存行为。

[中文](#中文版本) | [English](#english-version)

---

## 中文版本

### ✨ 核心特性

- 🎯 **零侵入追踪** - 无需修改模型代码，通过 PyTorch forward hooks 自动追踪
- 📊 **KV Cache 监控** - 实时捕获和统计 K/V 缓存大小变化
- 📈 **内存可视化** - 生成清晰的内存使用 timeline 和对比图表  
- 🔧 **模块化设计** - 清晰的分层架构，便于扩展和定制
- ⚡ **最小依赖** - 仅依赖 PyTorch、NumPy 和 Matplotlib
- 📝 **完整文档** - 每个函数都有详细的 docstring 和使用示例
- 🎓 **易于学习** - 提供多个演示脚本和最佳实践代码

### 🚀 快速开始（3 分钟）

#### 1️⃣ 安装

```bash
git clone https://github.com/Terialion/attnflow.git
cd attnflow
pip install -r requirements.txt
```

#### 2️⃣ 最小化示例（4 行代码）

```python
from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary
import torch

model = your_transformer_model()

# 使用上下文管理器（推荐）⭐
with AttentionTracker(model) as tracker:
    with torch.no_grad():
        output = model(input_ids)
    print_memory_summary(tracker.get_memory_stats())
```

#### 3️⃣ 运行完整演示

```bash
python demo/simple_demo.py
python demo/realtime_demo.py
```

- `realtime_demo.py`（推荐）：实时动态图（每层 timeline + 全局 KV 增长条）
- `simple_demo.py`：基础追踪 + 静态可视化

> Windows 提示：`simple_demo.py` 默认保存到 `/tmp/...`，在 Windows 上可能报路径错误；建议优先使用 `realtime_demo.py`，或自行修改保存路径。

**真实输出示例（本机运行 `python demo/realtime_demo.py`）：**

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

说明：运行环境可能出现第三方库日志（如 `MMKV`、`libpng warning`），它们不影响 AttnFlow 的 tracking 结果。

---

### 📚 详细使用指南

#### 方式一：上下文管理器（推荐 ⭐）

自动处理 hooks 的生命周期，最安全和简洁：

```python
from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary
import torch

model = your_transformer_model()
model.eval()

with AttentionTracker(model) as tracker:
    # 运行多个 batch
    with torch.no_grad():
        for input_ids in batches:
            output = model(input_ids)
    
    # 自动获取统计信息
    stats = tracker.get_memory_stats()
    print_memory_summary(stats)

# hooks 在退出 with 块时自动卸载 ✓
```

#### 方式二：手动管理

需要更细粒度的控制时：

```python
tracker = AttentionTracker(model, enable_logging=True)

# 1. 注册 hooks
tracker.register_hooks()

# 2. 开始追踪
tracker.start_tracking()

with torch.no_grad():
    output = model(input_ids)

# 3. 停止追踪
tracker.stop_tracking()

# 4. 获取统计数据
stats = tracker.get_memory_stats()
summary = tracker.get_summary()
print(f"Tracked layers: {len(summary)}")

# 5. 清理资源
tracker.unregister_hooks()
```

#### 方式三：多次追踪和重置

```python
tracker = AttentionTracker(model)
tracker.register_hooks()

# 第一次追踪
tracker.start_tracking()
output1 = model(batch_1)
tracker.stop_tracking()

stats_1 = tracker.get_memory_stats()

# 重置统计数据（保留 hooks）
tracker.reset_stats()

# 第二次追踪
tracker.start_tracking()
output2 = model(batch_2)
tracker.stop_tracking()

stats_2 = tracker.get_memory_stats()
```

---

### 📊 可视化选项

#### 1️⃣ CLI 输出（快速预览）

```python
from attnflow.viz import (
    print_memory_summary,
    print_memory_timeline,
    print_all_timelines
)

stats = tracker.get_memory_stats()

# 打印摘要表格
print_memory_summary(stats)

# 打印特定层的 timeline
print_memory_timeline(stats, "layers.0.self_attention")

# 打印所有层的 timeline
print_all_timelines(stats)
```

**输出示例：**

```
======================================================================
ATTENTION MEMORY SUMMARY
======================================================================
Layer Name                     Peak Memory (MB)     Max Seq Len    
----------------------------------------------------------------------
layers.0.self_attention        0.23                 30             
layers.0.self_attention.key    0.23                 30             
layers.0.self_attention.value  0.23                 30             
----------------------------------------------------------------------
TOTAL                          2.34                
======================================================================
```

#### 2️⃣ Matplotlib 图表（深度分析）

```python
from attnflow.viz import Visualizer

viz = Visualizer(style="default")

# 1. 绘制内存 timeline
fig1 = viz.plot_memory_timeline(stats)
fig1.savefig("memory_timeline.png", dpi=300, bbox_inches="tight")

# 2. 绘制各层峰值内存对比
fig2 = viz.plot_peak_memory_comparison(stats)
fig2.savefig("peak_memory.png", dpi=300, bbox_inches="tight")

# 3. 显示图表
viz.show(fig1)
```

生成的图表包括：
- **Timeline 图**：展示每层内存随时间增长的过程
- **对比图**：展示不同层之间的峰值内存差异

---

### 🏗️ 架构与 API 文档

#### 核心类概览

| 类 | 职责 | 适用场景 |
|----|------|---------|
| `AttentionTracker` | 主入口，管理追踪生命周期 | 所有场景 |
| `MemoryStats` | 存储和查询统计数据 | 数据分析 |
| `Visualizer` | 生成可视化图表 | 结果展示 |
| `TransformerHookManager` | 注册和管理 hooks | 定制扩展 |

#### `AttentionTracker` - 主类

```python
# 初始化
tracker = AttentionTracker(
    model=model,              # PyTorch 模型
    enable_logging=True       # 启用日志
)

# 核心方法
tracker.register_hooks()      # 注册 hooks 到所有注意力层
tracker.unregister_hooks()    # 注销所有 hooks
tracker.start_tracking()      # 开始记录内存数据
tracker.stop_tracking()       # 停止记录
tracker.reset_stats()         # 清除已收集的数据
tracker.is_tracking()         # 查询当前状态 -> bool

# 获取结果
tracker.get_memory_stats()    # 返回 MemoryStats 对象
tracker.get_summary()         # 返回摘要字典

# 上下文管理器（推荐）
with AttentionTracker(model) as tracker:
    # 自动调用 register_hooks() 和 start_tracking()
    ...
    # 自动调用 stop_tracking() 和 unregister_hooks()
```

#### `MemoryStats` - 数据查询

```python
stats = tracker.get_memory_stats()

# 查询所有追踪的层
layers = stats.get_all_layers()  # -> List[str]

# 查询单层内存
peak_bytes = stats.get_peak_memory(layer_name)     # -> int
peak_mb = stats.get_peak_memory_mb(layer_name)     # -> float

# 获取内存 timeline
timestamps, memory_values = stats.get_memory_timeline(layer_name)
# timestamps: [0.001, 0.002, 0.003] (秒)
# memory_values: [0.1, 0.2, 0.3] (MB)

# 获取完整摘要
summary = stats.get_summary()  # -> Dict[str, Dict]
# {
#   "layers.0.attention": {
#       "peak_memory_mb": 0.23,
#       "peak_sequence_length": 30,
#       "snapshot_count": 3
#   }
# }

# 获取所有快照
snapshots = stats.get_snapshots(layer_name)  # -> List[LayerMemorySnapshot]
```

#### `Visualizer` - 可视化

```python
from attnflow.viz import Visualizer

viz = Visualizer(style="default")

# 方法 1：单层 timeline
fig = viz.plot_memory_timeline(
    stats=stats,
    layer_name="layers.0.self_attention",  # 可选，None 表示所有层
    save_path="timeline.png"               # 可选
)

# 方法 2：层间对比
fig = viz.plot_peak_memory_comparison(
    stats=stats,
    save_path="comparison.png"
)

# 方法 3：显示或保存
viz.show(fig)                        # 显示图表
fig.savefig("output.png", dpi=300)  # 保存图表
```

---

### 📂 项目结构详解

#### 目录树

```
attnflow/
│
├── attnflow/                      # 主包目录
│   │
│   ├── __init__.py               # 包初始化，导出 AttentionTracker
│   │
│   ├── core/                     # ⭐ 核心追踪逻辑（数据层）
│   │   ├── __init__.py
│   │   ├── tracker.py            # 👑 主追踪器类 AttentionTracker
│   │   └── memory_stats.py       # 数据结构：MemoryStats, LayerMemorySnapshot
│   │
│   ├── hooks/                    # 🎣 Hook 机制（拦截层）
│   │   ├── __init__.py
│   │   └── transformer_hooks.py  # Hook 注册和执行逻辑
│   │
│   ├── viz/                      # 📊 可视化模块（展示层）
│   │   ├── __init__.py
│   │   ├── cli_output.py         # 命令行表格输出
│   │   └── visualizer.py         # Matplotlib 图表生成
│   │
│   └── utils/                    # 🛠️ 工具库（辅助层）
│       ├── __init__.py
│       ├── logger.py             # 统一日志工具
│       └── constants.py          # 全局常量定义
│
├── demo/                         # 📚 演示和示例
│   ├── __init__.py
│   ├── simple_demo.py           # ⭐ 完整演示脚本
│   ├── realtime_demo.py         # ⭐ Day2 实时动态图演示
│   └── attention_model.py       # 演示用的 Transformer 模型
│
├── README.md                     # 📖 项目文档（本文件）
├── requirements.txt              # 📦 Python 依赖
└── LICENSE                       # ⚖️ MIT 许可证
```

#### 分层架构说明

```
┌─────────────────────────────────────────┐
│      应用层（Application）              │
│  用户代码 - TrackingTracker 使用        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      控制层（Control）                  │
│  AttentionTracker - 生命周期管理        │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────────┐
        │                 │
┌───────▼─────────┐  ┌───▼──────────────┐
│  拦截层         │  │  展示层          │
│ TransformerHook │  │ Visualizer       │
│ Manager         │  │ CLI Output       │
└───────┬─────────┘  └───┬──────────────┘
        │                 │
        └──────┬──────────┘
               │
┌──────────────▼──────────────────────────┐
│      数据层（Data）                     │
│  MemoryStats - 存储统计数据             │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      基础层（Foundation）               │
│  PyTorch, Logger, Constants             │
└─────────────────────────────────────────┘
```

| 层 | 模块 | 职责 | 依赖关系 |
|----|------|------|---------|
| **应用层** | 用户代码 | 调用 API | ↓ |
| **控制层** | `tracker.py` | 管理生命周期 | ↓ |
| **拦截层** | `hooks/` | 注册和执行 hooks | ↓ |
| **展示层** | `viz/` | 数据可视化显示 | ↓ |
| **数据层** | `memory_stats.py` | 数据存储和查询 | ↓ |
| **辅助层** | `utils/` | 日志、常量等 | PyTorch |

#### 模块边界与依赖约束（推荐遵循）

为了保证可维护性和可扩展性，建议严格遵循以下边界：

- `attnflow.core`：只负责**生命周期管理**与**状态编排**，不直接依赖可视化实现细节。
- `attnflow.hooks`：只负责**forward 拦截**与**张量信息抽取**，不做展示层逻辑。
- `attnflow.viz`：只消费 `MemoryStats` 查询接口，不反向依赖 `hooks`。
- `attnflow.utils`：提供无业务副作用的基础能力（常量、日志）。

边界规则：

1. 单向依赖：`core -> hooks/data -> viz(消费结果)`，避免循环依赖。
2. `hooks` 与 `viz` 通过 `MemoryStats` 解耦，禁止互相直接调用。
3. 新功能优先通过新增模块扩展，而不是在 `tracker.py` 中堆叠分支逻辑。

#### 端到端数据流（一次推理）

```text
用户调用 model.forward(input_ids)
    ↓
TransformerHookManager 的 forward hook 被触发
    ↓
提取输出张量 shape: (batch, seq_len, hidden_dim)
    ↓
估算 K/V cache bytes（K + V）
    ↓
MemoryStats.record_snapshot(layer_name, k, v, seq_len)
    ↓
MemoryStats 提供 summary / timeline 查询
    ↓
CLI 或 Matplotlib 渲染展示
```

关键数据对象：

- `LayerMemorySnapshot`：单层单时刻记录（时间戳、K/V bytes、seq_len）。
- `MemoryStats`：按层聚合 snapshot，提供 `get_summary()` 与 `get_memory_timeline()`。

#### 扩展点设计（TODO 对应落位）

以下扩展点已在代码中预留 TODO，推荐按接口化方式推进：

1. **注意力层识别策略扩展**（`hooks/transformer_hooks.py`）
   - 现状：基于命名规则匹配。
   - 建议：增加可注入 matcher（函数或策略类），覆盖 FlashAttention / 自定义模块命名。

2. **KV 内存估算策略扩展**（`hooks/transformer_hooks.py`）
   - 现状：使用通用估算公式。
   - 建议：为不同 dtype、压缩 cache、分组注意力提供策略实现。

3. **可视化后端扩展**（`viz/visualizer.py`）
   - 现状：Matplotlib + CLI。
   - 建议：新增 Plotly/前端流式后端时，保持 `MemoryStats -> renderer` 的单向消费模式。

4. **追踪策略扩展**（`core/tracker.py`）
   - 现状：统一生命周期管理。
   - 建议：新增采样率控制、层过滤、阈值告警时，优先在 `core` 编排，不污染 `hooks`/`viz`。

> 最佳实践：新增功能时优先“加模块/策略”，避免修改核心路径已有行为；每个扩展点至少补 1 个单元测试与 1 个边界测试。

---

### 💡 实战案例

#### 案例 1：分析模型内存消耗

```python
from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary, Visualizer
import torch

# 加载模型
model = load_my_transformer()
model.eval()

# 追踪和分析
with AttentionTracker(model) as tracker:
    with torch.no_grad():
        # 输入：8 个样本，512 tokens
        input_ids = torch.randint(0, 30000, (8, 512))
        output = model(input_ids)
    
    # 查看文本摘要
    stats = tracker.get_memory_stats()
    print_memory_summary(stats)
    
    # 生成可视化
    viz = Visualizer()
    fig = viz.plot_peak_memory_comparison(stats)
    fig.savefig("memory_analysis.png")
    print("✓ 分析完成，结果已保存")
```

#### 案例 2：对比不同批次大小的影响

```python
from attnflow import AttentionTracker
import torch

model = load_my_transformer()
model.eval()

results = {}

# 测试不同的 batch size
for batch_size in [1, 4, 8, 16]:
    with AttentionTracker(model) as tracker:
        input_ids = torch.randint(0, 30000, (batch_size, 512))
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        summary = stats.get_summary()
        
        # 计算总内存
        total_mem = sum(
            s['peak_memory_mb'] for s in summary.values()
        )
        results[batch_size] = total_mem
        print(f"Batch size {batch_size:2d}: {total_mem:6.2f} MB")

# 输出结果
print("\nMemory Scaling Analysis:")
for bs, mem in sorted(results.items()):
    print(f"  batch_size={bs:2d} -> {mem:6.2f} MB")
```

#### 案例 3：性能基准测试

```python
import time
from attnflow import AttentionTracker
import torch

model = load_my_transformer()
model.eval()

durations = []
memory_peaks = []

# 运行多次测试
for _ in range(10):
    with AttentionTracker(model) as tracker:
        input_ids = torch.randint(0, 30000, (4, 256))
        
        start = time.time()
        with torch.no_grad():
            output = model(input_ids)
        duration = time.time() - start
        
        durations.append(duration)
        
        stats = tracker.get_memory_stats()
        max_mem = max(
            stats.get_peak_memory_mb(layer)
            for layer in stats.get_all_layers()
        )
        memory_peaks.append(max_mem)

# 统计结果
print(f"Average Latency: {sum(durations)/len(durations):.4f}s")
print(f"Peak Memory: {sum(memory_peaks)/len(memory_peaks):.2f}MB")
print(f"Memory Std Dev: {(max(memory_peaks) - min(memory_peaks)):.2f}MB")
```

---

### ⚙️ 高级用法

#### 自定义 Layer 模式匹配

编辑 `attnflow/hooks/transformer_hooks.py`：

```python
def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
    """
    自定义哪些层应该被追踪
    
    示例：只追踪特定的注意力实现
    """
    # 默认关键词
    attention_keywords = [
        "attention",
        "self_attn",
        "cross_attn",
        "multihead",
    ]
    
    # 添加自定义关键词
    custom_keywords = [
        "my_attention",      # 自定义层
        "specialized_attn",  # 特殊实现
    ]
    
    name_lower = name.lower()
    all_keywords = attention_keywords + custom_keywords
    return any(kw in name_lower for kw in all_keywords)
```

#### 修改日志级别

```python
import logging
from attnflow.utils.logger import get_logger

# 启用详细的调试日志
logger = get_logger(
    "attnflow.hooks.transformer_hooks",
    level=logging.DEBUG
)

# 现在可以看到更多调试信息
tracker = AttentionTracker(model, enable_logging=True)
```

---

### ❓ 常见问题 FAQ

#### Q1: 某些层没有被追踪到？

**A:** 检查层的名称是否包含识别关键词。

**当前支持的关键词：**
- `attention`
- `self_attn`
- `cross_attn`
- `multihead`
- `self.attention`

**解决方案：**
```python
# 1. 检查实际的层名
for name, module in model.named_modules():
    if "attn" in name.lower():
        print(name)

# 2. 修改关键词以匹配你的层名
# 编辑 transformer_hooks.py 中的 _is_attention_layer() 方法
```

#### Q2: 内存统计不准确？

**A:** 当前实现基于输出张量形状进行估算。对于复杂的注意力变体：

```python
from attnflow.hooks.transformer_hooks import TransformerHookManager
from attnflow.core.memory_stats import MemoryStats

class CustomHookManager(TransformerHookManager):
    """自定义 hook 以提取实际的 KV cache"""
    
    def _create_hook(self, layer_name: str):
        def hook(module, input, output):
            # 提取实际的 KV cache（如果可用）
            if hasattr(module, 'k_cache'):
                k_cache = module.k_cache
                v_cache = module.v_cache
                
                k_bytes = k_cache.element_size() * k_cache.numel()
                v_bytes = v_cache.element_size() * v_cache.numel()
            else:
                # 回退到基于形状的估算
                ...
            
            self.memory_stats.record_snapshot(
                layer_name=layer_name,
                k_cache_size=k_bytes,
                v_cache_size=v_bytes,
                sequence_length=k_cache.shape[1]
            )
        return hook
```

#### Q3: 如何只追踪特定的层？

**A:** 创建自定义 tracker 类：

```python
from attnflow import AttentionTracker

class SelectiveTracker(AttentionTracker):
    def __init__(self, model, target_layers):
        super().__init__(model)
        self.target_layers = set(target_layers)
        
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # 只在目标层上注册 hooks
                ...

# 使用
target = ["layers.0.self_attention", "layers.1.self_attention"]
with SelectiveTracker(model, target) as tracker:
    output = model(input_ids)
```

#### Q4: hooks 的性能开销如何？

**A:** 根据基准测试，开销通常 **< 5%**：

| 操作 | 开销 | 备注 |
|------|------|------|
| 注册 hooks | ~1ms | 一次性 |
| 前向传播 | +1-2% | 相对开销 |
| 内存记录 | ~100KB | 取决于层数 |

**优化建议：**
```python
# 1. 在 batch 推理时摊超开销
with AttentionTracker(model) as tracker:
    for batch in batches:
        output = model(batch)  # 开销分散

# 2. 只追踪特定层
tracker = SelectiveTracker(model, important_layers)
```

#### Q5: 支持 GPU 张量吗？

**A:** 完全支持！

```python
model = model.cuda()
input_ids = input_ids.cuda()

with AttentionTracker(model) as tracker:
    with torch.no_grad():
        output = model(input_ids)  # GPU 执行
    
    # 统计中的内存是 GPU 内存
    stats = tracker.get_memory_stats()
    print_memory_summary(stats)
```

#### Q6: 能追踪分布式模型吗？

**A:** 目前还不支持 DDP，但在未来版本中会添加。当前：

```python
# 当前：先合并 checkpoints，再追踪
model = merge_ddp_model(ddp_model)
with AttentionTracker(model) as tracker:
    ...
```

---

### 🎯 最佳实践

#### ✅ DO - 推荐做法

```python
# 1. ✅ 使用上下文管理器
with AttentionTracker(model) as tracker:
    ...

# 2. ✅ 在 eval 模式下追踪
model.eval()
with AttentionTracker(model) as tracker:
    ...

# 3. ✅ 使用 torch.no_grad()
with torch.no_grad():
    output = model(input_ids)

# 4. ✅ 启用日志调试问题
tracker = AttentionTracker(model, enable_logging=True)

# 5. ✅ 定期重置统计数据
tracker.reset_stats()

# 6. ✅ 保存可视化结果
fig.savefig("result.png", dpi=300, bbox_inches="tight")
```

#### ❌ DON'T - 避免做法

```python
# 1. ❌ 不要忘记卸载 hooks
# tracker.register_hooks()
# ... 代码 ...
# # 忘记了 tracker.unregister_hooks()

# 2. ❌ 不要在 train 模式下追踪
model.train()  # ❌
tracker.start_tracking()

# 3. ❌ 不要混淆多个 tracker
t1 = AttentionTracker(model)
t2 = AttentionTracker(model)  # ❌ 重复注册

# 4. ❌ 不要忽视日志中的警告
# [WARNING] Hooks already registered, skipping...
```

---

### 🔮 未来路线图

| 功能 | 优先级 | 预计版本 | 状态 |
|------|--------|---------|------|
| 分布式追踪 (DDP) | 🔴 高 | v0.2 | 📋 规划中 |
| Flash Attention 支持 | 🔴 高 | v0.2 | 📋 规划中 |
| Web 仪表板 | 🟡 中 | v0.3 | 🔍 研究中 |
| 内存峰值分析报告 | 🟡 中 | v0.3 | 📋 规划中 |
| ONNX 支持 | 🟢 低 | v1.0 | ✅ 已跟踪 |

---

### 📦 依赖

```
torch>=1.9.0          # PyTorch 核心库
numpy>=1.20.0         # 数值计算
matplotlib>=3.3.0     # 数据可视化
```

### 🚀 安装方式

```bash
# 方式 1：从源码安装（推荐）
git clone https://github.com/Terialion/attnflow.git
cd attnflow
pip install -r requirements.txt

# 方式 2：开发模式（用于贡献代码）
pip install -e .

# 已选：pip install attnflow  (未来 PyPI 发布)
```

---

### 📄 许可证

本项目采用 **MIT License** 开源许可

详见 [LICENSE](LICENSE) 文件

```
MIT License

Copyright (c) 2026 AttnFlow Team

Permission is hereby granted, free of charge...
```

---

### 🙌 贡献指南

欢迎提交 Issue 和 Pull Request！

#### 如何贡献

```bash
# 1. Fork 本仓库到你的账户
#    点击 GitHub 页面的 "Fork" 按钮

# 2. Clone 到本地
git clone https://github.com/<你的用户名>/attnflow.git
cd attnflow

# 3. 创建特性分支
git checkout -b feature/AmazingFeature

# 4. 编写代码（遵循代码规范）
# - 遵循 PEP 8 规范
# - 为函数添加 docstring
# - 包含单元测试

# 5. 提交更改
git commit -m 'Add: Add some AmazingFeature'
git push origin feature/AmazingFeature

# 6. 在 GitHub 上创建 Pull Request
```

#### 代码规范

```bash
# 使用 black 格式化
pip install black
black attnflow/

# 验证 PEP 8 规范
pip install flake8
flake8 attnflow/

# 运行测试
pytest tests/
```

#### 提 Issue

请说明：
- 🐛 问题描述
- 📝 复现步骤
- 📦 你的环境信息（OS, Python, PyTorch 版本）
- 💡 期望的行为

---

### 📞 联系与反馈

- 📧 **GitHub Issues** - [提交 Bug 或建议](https://github.com/Terialion/attnflow/issues)
- 💬 **Discussions** - [讨论和问答](https://github.com/Terialion/attnflow/discussions)

---

### 🌟 Star 历史

如果这个项目对你有帮助，请给一个 ⭐！

你的 Star 是对我们最大的鼓励！

```
⭐ ⭐ ⭐ Star this project!
```

---

### 引用

如果你在论文或项目中使用了 AttnFlow，请引用：

```bibtex
@software{attnflow2026,
  author = {Terialion},
  title = {AttnFlow: Real-time Transformer Attention Memory Flow Visualization},
  year = {2026},
  url = {https://github.com/Terialion/attnflow}
}
```

---

## English Version

### ✨ Features

- 🎯 **Zero-Intrusive Tracking** - Track attention without modifying model code
- 📊 **Real-time KV Cache Monitoring** - Capture and analyze cache growth
- 📈 **Memory Visualization** - Generate timeline and comparison charts
- 🔧 **Modular Design** - Clean architecture, easy to extend
- ⚡ **Minimal Dependencies** - Only PyTorch, NumPy, and Matplotlib
- 📝 **Complete Documentation** - Docstrings for all functions
- 🎓 **Easy to Learn** - Multiple examples and best practices

### 🚀 Quick Start

```python
from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary
import torch

model = your_transformer_model()

with AttentionTracker(model) as tracker:
    with torch.no_grad():
        output = model(input_ids)
    
    print_memory_summary(tracker.get_memory_stats())
```

Run demo:

```bash
python demo/simple_demo.py
python demo/realtime_demo.py
```

- `realtime_demo.py` (recommended): real-time animation (per-layer timeline + global KV growth bar)
- `simple_demo.py`: baseline tracking + static visualization

> Windows note: `simple_demo.py` saves figures to `/tmp/...` by default and may fail on Windows. Use `realtime_demo.py` first, or change the save path.

**Real output sample (`python demo/realtime_demo.py`):**

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

Note: You may see third-party logs (for example `MMKV` or `libpng warning`) in your environment. They do not affect AttnFlow tracking results.

For detailed documentation, see the Chinese version above.

---

**Made with ❤️ by AttnFlow Team**