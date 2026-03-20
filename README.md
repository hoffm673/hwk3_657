# HW3 — README

## Overview

This homework consists of three notebooks covering JIT compilation in JAX and PyTorch.

| Notebook | Problem | Purpose |
|---|---|---|
| `hw_3_657_Conceptual.ipynb` | Problem 1 | Conceptual answers and illustrative code examples — **do not run** |
| `jax_jit_analysis_1_.ipynb` | Problem 2 | JAX JIT benchmarking and retrace analysis |
| `torch_compile_analysis.ipynb` | Problem 3 | `torch.compile` backend benchmarking and graph inspection |

---

## Problem 1 — `hw_3_657_Conceptual.ipynb`

**This notebook does not need to be run.** It contains written answers and code snippets that serve as illustrative examples only. No output is expected or required.

---

## Problem 2 — `jax_jit_analysis_1_.ipynb`

**Environment:** Google Colab with a **T4 GPU** is strongly recommended. Set *Runtime > Change runtime type > T4 GPU* before running.

**Dependencies** (installed automatically in the first cell):
```
jax  jaxlib  matplotlib  numpy
```

**Run order:** Execute all cells top to bottom in order.

**Important fix — `process_batch` decorator:**

The decorator requires `functools.partial` because `jax.jit` does not support keyword-only decorator syntax directly:

```python
# WRONG — raises TypeError: jit() missing 1 required positional argument: 'fun'
@jax.jit(static_argnums=(1,))

# CORRECT
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def process_batch(data, batch_size):
    ...
```

**What each part does:**

- **Part 1:** Benchmarks eager vs. JIT first call vs. JIT cached call across matrix sizes `[100, 500, 1000, 5000]`. Produces an absolute timing chart and a speedup/overhead bar chart.
- **Part 2:** Demonstrates JAX retracing — shows that passing a new input shape triggers recompilation with high upfront cost, while repeated calls on the same shape use the cached XLA binary. Saves output to `jit_timing.png`.
- **Part 3:** Switches to PyTorch (`torch.jit.script`). Benchmarks eager vs. compiled execution and uses the PyTorch profiler to compare kernel counts and memory throughput.

---

## Problem 3 — `torch_compile_analysis.ipynb`

**Environment:** Google Colab with a **T4 GPU** is strongly recommended. Set *Runtime > Change runtime type > T4 GPU* before running.

**Dependencies** (installed automatically in the first cell):
```
torch  torchvision  matplotlib  numpy
```

**Run order:** Execute all cells top to bottom in order.

**Notes:**

- **`cudagraphs` backend** is automatically skipped if no GPU is detected — this is expected behavior, not an error.
- **`symbolic_trace` cell** will print `"Hello"` to stdout during tracing. This is intentional — it demonstrates that non-tensor Python side effects (like `print`) execute during tracing and are not captured in the FX graph.
- **Compile time output** shows multiple runtimes per function (e.g., `0.1450, 0.0110`). These reflect repeated compilation passes by TorchDynamo and are normal.
- If you see `[warn] cudagraphs: stream still in capture mode after warmup`, CUDA graph capture did not finalize cleanly. This is a known issue on some Colab configurations and does not affect the other backends.

**What each part does:**

- **Part 1:** Benchmarks `SimpleModel` (3-layer MLP, input 512 → hidden 1024 → output 256) across `eager`, `inductor`, and `cudagraphs` backends. Reports forward+backward pass time in milliseconds.
- **Later parts:** Inspects TorchDynamo guard lists to show what conditions a compiled graph checks before reuse. Uses `torch.fx.symbolic_trace` to print the FX graph of a sample function, demonstrating which operations are captured (tensor ops) and which are not (Python side effects like `print` and list mutation).
- 
