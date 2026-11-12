# PyTorch

## Tensor Basics

Q1: ***What are the differences between a PyTorch Tensor and a NumPy ndarray? How do you convert between them?***

- **Device**: tensors can live on accelerators such as GPU/MPS (`.to("cuda")`); ndarrays are CPU-only.
- **Autograd**: tensors can have `requires_grad=True` and record a computational graph; ndarrays cannot.
- **Default dtype**: `torch.tensor([1.0])` defaults to `float32`, `np.array([1.0])` to `float64` — data coming from NumPy often silently becomes float64, causing dtype mismatches with model weights or slowdowns.
- Conversion:
    - `torch.from_numpy(a)` and `t.numpy()`: **share memory** (zero-copy); modifying one changes the other; CPU tensors only.
    - `torch.tensor(a)`: **always copies**.
    - GPU tensors or tensors requiring grad need `t.detach().cpu().numpy()` first.
- The API style is similar (broadcasting, indexing, `sum(dim=...)` corresponds to `axis=...`), but PyTorch also has in-place variants (trailing underscore, `add_`).

Q2: ***What is the difference between view, reshape, transpose and contiguous? When do you get a view and when a copy?***

- A tensor = an underlying **storage** + `shape` + `stride` + `offset`. A **view** shares the storage and only changes the metadata.
- **View ops**: `view`, `transpose`/`t`, `permute`, slicing `x[:, 1:]`, `expand`, `squeeze/unsqueeze`, `narrow`, `detach`.
- After `transpose/permute` the strides are no longer row-major and the tensor is **non-contiguous** (`is_contiguous()==False`); `view` then raises an error, so use `.contiguous().view(...)` (copies) or simply `reshape`.
- `reshape`: returns a view when possible, otherwise a copy — so **do not rely on whether it shares memory**.
- **Copy ops**: `clone()`, `contiguous()` (when non-contiguous), advanced indexing `x[idx_tensor]`, boolean indexing `x[mask]`, `torch.tensor(...)`.
- `expand` allocates no memory (stride 0), but writing to it in place is wrong; use `repeat` when you need a real copy.

```python
a = torch.arange(6).view(2, 3)
b = a.t()                    # view, shares storage
b.is_contiguous()            # False
# b.view(-1)                 # RuntimeError
b.reshape(-1)                # copies internally: tensor([0, 3, 1, 4, 2, 5])
a.data_ptr() == b.data_ptr() # True
```

Q3: ***What are PyTorch's broadcasting rules? What are the common pitfalls?***

- Align dimensions starting from the **last one**; each dimension must be equal, or one of them is 1, or one of them is missing (treated as 1); the result takes the larger size.
- Examples: `(B, T, C) + (C,)` → OK; `(B, 1, T) * (B, T, 1)` → `(B, T, T)`.
- Common pitfalls:
    - `y_pred` of shape `(N, 1)` and `y` of shape `(N,)`: `(y_pred - y)` broadcasts to `(N, N)` and MSE is silently wrong — `nn.MSELoss` warns about this.
    - In-place ops cannot change the left-hand shape: `a.add_(b)` requires `b` to broadcast to `a`.
    - Broadcasting can silently create huge tensors that fill GPU memory (e.g. pairwise distances `(N,1,D)-(1,M,D)`).
- Habit: add `assert x.shape == (...)` at key points, or use `keepdim=True` to keep dimensions.

## Autograd

Q4: ***Explain how autograd and the computational graph work***

- PyTorch uses a **dynamic graph** (define-by-run): during the forward pass, every op on a `requires_grad` tensor creates a `grad_fn` node (e.g. `MulBackward0`) and saves the tensors needed for backward (`ctx.save_for_backward`), forming a DAG.
- `loss.backward()` performs **reverse-mode automatic differentiation** (vector-Jacobian products) from the loss in **reverse topological order**, accumulating gradients into the `.grad` of leaf tensors.
- After backward, the graph and intermediate buffers are **freed** by default; to backprop through it again pass `retain_graph=True` (e.g. several losses sharing part of the graph with separate backward calls).
- Calling `backward` on a non-scalar output requires a `gradient` argument (the vector $v$ in the VJP), e.g. `y.backward(torch.ones_like(y))`.
- `torch.autograd.grad(outputs, inputs, create_graph=True)` returns gradients directly without writing `.grad`; `create_graph=True` enables higher-order derivatives (e.g. gradient penalty, MAML).
- Because the graph is dynamic, Python control flow (if / for / recursion) participates directly in graph construction, and the graph can differ every iteration.

Q5: ***What are requires_grad, torch.no_grad, torch.inference_mode and detach? How do they differ?***

- `requires_grad=True`: marks a tensor as needing gradients; ops on it are recorded. `nn.Parameter` defaults to True. Freeze a parameter with `p.requires_grad_(False)`.
- `torch.no_grad()`: context manager; ops inside **build no graph** and outputs have `requires_grad=False`; used for validation, EMA updates and manual weight edits.
- `torch.inference_mode()`: more aggressive than `no_grad`, also disabling view tracking and version counters, so it is faster; but the resulting **inference tensors cannot later participate in autograd** (cannot be used in computations that need gradients). Preferred for pure inference/deployment.
- `x.detach()`: returns a new tensor that **shares storage** but is cut off from the graph (`requires_grad=False`); used to stop gradients (target networks, stop-gradient, logging loss values). Note that modifying the detached tensor also modifies the original.
- `torch.set_grad_enabled(mode)`: toggles conditionally, commonly used when one function serves both train and val.
- **Orthogonal** to `model.eval()`: `eval()` changes Dropout/BN behavior but does not disable gradients; use both at inference.

Q6: ***What is a leaf tensor? Why do gradients accumulate? Why call zero_grad every step?***

- **Leaf tensors**: tensors created directly by the user (with `requires_grad=True`), or any tensor with `requires_grad=False`; their `grad_fn is None`. Only leaf tensors get `.grad` populated by `backward` by default; non-leaf tensors need `.retain_grad()`.
- `.grad` is **accumulated** (`+=`) rather than overwritten: this design supports a parameter used multiple times in the graph, multiple losses backpropagated separately, and gradient accumulation.

```python
w = torch.ones(3, requires_grad=True)
for _ in range(2):
    (w * 2).sum().backward()
print(w.grad)   # tensor([4., 4., 4.]), accumulated twice
```

- Therefore clear gradients before each optimizer step: `optimizer.zero_grad(set_to_none=True)` (the default in PyTorch 2.x) sets `.grad` to `None` instead of filling zeros, which saves memory and is faster; but code reading `.grad` afterwards must handle `None`.
- Pitfall: taking gradients of a tensor after `.cuda()` or `.to()` — `w = torch.randn(3, requires_grad=True).cuda()` is a **non-leaf**, so `w.grad` is None. Use `torch.randn(3, device="cuda", requires_grad=True)` instead.

Q7: ***How do you implement a custom autograd Function? When is it needed?***

- Subclass `torch.autograd.Function`, implement static methods `forward(ctx, ...)` and `backward(ctx, grad_out)`, save tensors with `ctx.save_for_backward`, and call it via `.apply`.
- Use cases: more numerically stable or memory-efficient gradient formulas, calling custom CUDA/Triton kernels, straight-through estimators (quantization), gradient reversal layers (GRL).

```python
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = torch.sigmoid(x)
        ctx.save_for_backward(x, s)
        return x * s

    @staticmethod
    def backward(ctx, grad_out):
        x, s = ctx.saved_tensors
        return grad_out * (s + x * s * (1 - s))   # one gradient per forward input

x = torch.randn(5, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(Swish.apply, (x,))   # True: verifies backward with finite differences
```

- Key points: `backward` returns as many gradients as `forward` has inputs, `None` for non-tensor inputs; verify with `gradcheck` (double precision); extra care is needed for `torch.compile` compatibility or higher-order derivatives (e.g. `setup_context`, `once_differentiable`).

Q8: ***What problems do in-place operations cause for autograd?***

- If a tensor needed by backward is modified in place, you get `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. Autograd detects this via each tensor's **version counter** (`_version`).
- Example: `y = x.sigmoid(); y.mul_(2)` — the backward of `sigmoid` needs its output `y`, which has been overwritten, so it fails.
- A leaf tensor with `requires_grad=True` cannot be modified in place directly (do it under `no_grad`, as optimizers do when updating parameters).
- Advice: avoid in-place ops unless memory pressure really demands it (`nn.ReLU(inplace=True)` is generally safe because ReLU's backward only needs the sign of its output).

## Modules and Data

Q9: ***What are parameters, buffers and submodules in an nn.Module?***

- `nn.Module` registers them automatically via `__setattr__`:
    - **Parameter** (`nn.Parameter`): learnable parameters, listed by `parameters()` and updated by the optimizer.
    - **Buffer** (`register_buffer`): non-learnable state such as BN's `running_mean`, RoPE's cos/sin cache, or a causal mask; moved by `.to(device)` and saved in `state_dict` by default; `persistent=False` excludes it from saving.
    - **Submodules**: other `nn.Module`s assigned as attributes, managed recursively.
- Pitfalls:
    - Submodules held in a Python `list`/`dict` are not registered (their parameters are missing from `parameters()` and not moved by `.to()`); use `nn.ModuleList` / `nn.ModuleDict` / `nn.ParameterList`.
    - A plain tensor attribute `self.mask = torch.ones(...)` is not moved by `.to("cuda")`; use `register_buffer`.
- Common APIs: `named_parameters()`, `modules()`, `apply(fn)` (e.g. custom initialization), `state_dict()`, `requires_grad_()`, `zero_grad()`.

Q10: ***What is the difference between model.train() and model.eval()?***

- They only recursively set each submodule's `self.training` flag, which affects **mode-dependent layers**:
    - **Dropout**: train randomly drops and scales by $1/(1-p)$; eval is the identity.
    - **BatchNorm**: train uses batch statistics and updates running stats; eval uses running stats.
- They do **not** disable gradient computation; at inference also use `torch.no_grad()` / `inference_mode()`.
- Common bugs: forgetting to switch back to `model.train()` after validation; fine-tuning BN in eval mode; train-mode BN is unstable with tiny batches, so freeze BN (call `eval()` on BN submodules and freeze their parameters).

Q11: ***How do Dataset and DataLoader work? What do num_workers, pin_memory, collate_fn and sampler do?***

- **Dataset**: map-style implements `__len__` and `__getitem__(i)`; iterable-style (`IterableDataset`) implements `__iter__`, suited to streaming data.
- **DataLoader** handles batching, shuffling and parallel loading:
    - `sampler` / `batch_sampler`: decides the index order. `shuffle=True` is just `RandomSampler`; use `WeightedRandomSampler` for class imbalance; use `DistributedSampler` for DDP (call `sampler.set_epoch(epoch)` every epoch, otherwise every epoch has the same shuffle order).
    - `collate_fn`: merges a list of samples into a batch; pad variable-length sequences and build masks here.
    - `num_workers`: number of subprocesses running `__getitem__` in parallel; 0 means loading in the main process. Combine with `persistent_workers=True` (avoids re-spawning processes every epoch) and `prefetch_factor`.
    - `pin_memory=True`: puts batches in **page-locked memory** so host-to-device copies are faster and can overlap with compute via `.to("cuda", non_blocking=True)`.
    - `drop_last`: drops the last incomplete batch (useful for BN or when fixed shapes are required).
- Pitfalls: with multiple workers, control each worker's random seed via `worker_init_fn` / `generator` (otherwise NumPy augmentations may repeat across workers); an `IterableDataset` must shard manually using `get_worker_info()`, otherwise data is duplicated; do not do CUDA work inside `__getitem__`.

```python
def collate(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    mask = torch.arange(x.size(1))[None, :] < lens[:, None]
    return x, mask, torch.tensor(ys)

loader = torch.utils.data.DataLoader(
    ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True,
    persistent_workers=True, collate_fn=collate, drop_last=True)
```

## Training Techniques

Q12: ***How does mixed-precision training work? What do autocast and GradScaler do? How do fp16 and bf16 differ?***

- `torch.autocast`: inside the context, precision is chosen per op — matmul/conv run in low precision, while numerically sensitive ops such as softmax, reductions and losses stay in fp32. Parameters (master weights) remain fp32.
- **fp16**: 5 exponent bits, small dynamic range, so small gradients **underflow to 0** → needs `GradScaler`: multiply the loss by a large factor before backward and unscale before `step`; if inf/nan is found, **skip the step** and lower the scale.
- **bf16**: 8 exponent bits (same dynamic range as fp32) with lower mantissa precision; generally **no GradScaler needed**. The first choice on newer hardware such as A100/H100.

```python
scaler = torch.amp.GradScaler("cuda")          # can be omitted for bf16
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                 # must unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                     # skipped if inf/nan
    scaler.update()
```

- Pitfalls: `backward` does not need to be inside autocast; forgetting `unscale_` before gradient clipping; mixing dtypes when computing the loss outside autocast.

Q13: ***How do you implement gradient accumulation and gradient clipping?***

- **Gradient accumulation**: when a large batch does not fit in memory, run $k$ micro-batches with backward but no step, relying on `.grad` accumulation, for an effective batch of $k\times$. Divide the loss **by $k$** to match the mean over the large batch.
- **Gradient clipping**: `clip_grad_norm_` rescales so the global L2 norm of all parameter gradients is at most `max_norm` ($g\leftarrow g\cdot\min(1,\frac{c}{\|g\|})$), preventing gradient explosion; 1.0 is common for Transformers. It returns the pre-clipping norm, useful for monitoring. `clip_grad_value_` clips element-wise and is less common.

```python
accum = 4
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accum
    loss.backward()
    if (i + 1) % accum == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

- Pitfalls: BN statistics are still per micro-batch, so it is not equivalent to a large batch; under DDP wrap non-sync steps in `with model.no_sync():` to avoid an all-reduce on every micro-batch; clip **after accumulation is complete and before step**; for token-weighted losses (variable-length sequences) normalize by the total token count rather than simply dividing by $k$.

Q14: ***What is the difference between DataParallel and DistributedDataParallel (DDP)?***

- **DataParallel (DP)**: single process, multiple threads; every step it **replicates** the model from the main GPU to all GPUs, scatters inputs, gathers outputs to the main GPU to compute the loss, then reduces gradients back to the main GPU. Limited by the **GIL**; main-GPU memory and communication become the bottleneck; single machine only. No longer recommended.
- **DDP**: **one process per GPU**, each holding a full model replica; parameters are broadcast at initialization; during backward, autograd hooks group gradients into **buckets** and **all-reduce** them (Ring/NCCL), **overlapping** with backward compute; each process runs its own `optimizer.step()`, keeping parameters in sync. Supports multiple machines.
- Usage essentials: `torchrun --nproc_per_node=N train.py`; `dist.init_process_group("nccl")`; `DistributedSampler` + `set_epoch`; save/print only on rank 0; save `model.module.state_dict()`.
- If some parameters are unused in the forward pass, set `find_unused_parameters=True` (extra overhead).
- Follow-up: model does not fit on one GPU → **FSDP** (PyTorch 2.x recommends `fully_shard`/FSDP2, sharding parameters, gradients and optimizer state, corresponding to ZeRO-3), tensor parallelism, pipeline parallelism.

Q15: ***What is torch.compile? What are its benefits and caveats?***

- Introduced in PyTorch 2.0: `model = torch.compile(model)`. **TorchDynamo** captures FX graphs at the Python bytecode level, **AOTAutograd** generates the backward graph ahead of time, and the **Inductor** backend fuses ops and generates Triton (GPU) / C++ (CPU) kernels.
- Benefits: fewer kernel launches and memory reads/writes (e.g. fusing chains of elementwise ops), speedups without code changes; `mode="reduce-overhead"` uses CUDA Graphs to cut small-batch overhead, `mode="max-autotune"` does more tuning.
- Caveats:
    - The first call compiles and is slow; **changing input shapes** triggers recompilation (set `dynamic=True` or pad/bucket shapes).
    - **Graph breaks**: unsupported Python code (prints, `.item()`, data-dependent control flow, some third-party libraries) splits the graph and falls back to eager, reducing the gains; diagnose with `TORCH_LOGS="graph_breaks"` or `torch._dynamo.explain`.
    - Results differ from eager at the floating-point level; when debugging, confirm correctness in eager first.

Q16: ***How do you save and load models? Why is state_dict recommended?***

- **Recommended**: save the `state_dict` (an ordered dict of parameters and buffers), which is decoupled from code and portable.
- `torch.save(model)` pickles the whole object and depends on class definition paths, so it breaks easily after refactoring and is a security risk. Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, allowing only safe types such as tensors.
- To resume training, also save the optimizer, scheduler, `GradScaler`, epoch/step and RNG states.

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
}
torch.save(ckpt, "ckpt.pt")

ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])            # with strict=True keys must match exactly
optimizer.load_state_dict(ckpt["optimizer"])
```

- Pitfalls: keys of DDP / `torch.compile`-wrapped models carry `module.` / `_orig_mod.` prefixes — save the underlying module or strip the prefixes when loading; use `map_location` to avoid loading onto the wrong GPU; call `model.to(device)` before creating the optimizer (or loading its state); for partial loading (transfer learning) use `strict=False` and check the returned `missing_keys` / `unexpected_keys`. Large models commonly use the `safetensors` format (safe, supports fast mmap loading).

Q17: ***How do you make experiments reproducible?***

- Fix all random seeds:

```python
import random, numpy as np, torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)              # seeds CPU and all CUDA devices
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
torch.backends.cudnn.benchmark = False              # do not auto-select (possibly different) conv algorithms
torch.use_deterministic_algorithms(True)            # raise on non-deterministic ops
# some cuBLAS ops also need the env var CUBLAS_WORKSPACE_CONFIG=:4096:8
```

- DataLoader: pass a seeded `generator` and seed each worker via `worker_init_fn`.
- Sources of non-determinism: cuDNN autotuning, backward of `atomicAdd`-style scatter/index_add, reduction order in multi-GPU communication, different hardware/drivers/library versions.
- Honest answer: **bitwise reproducibility** is only achievable on identical hardware and software stacks, at a cost in speed; the usual goal is statistical reproducibility (mean and variance over several seeds).

## Debugging and Performance

Q18: ***How do you debug a NaN/Inf loss during training?***

- **Locate**:
    - `torch.autograd.set_detect_anomaly(True)`: when backward produces NaN, it reports the corresponding forward op (very slow, debugging only).
    - Add `assert torch.isfinite(x).all()` in the forward pass, or register forward hooks that check every layer's output.
    - Log the gradient norm (the return value of `clip_grad_norm_`) to see at which step it explodes; inspect the offending batch.
- **Common causes and fixes**:
    - NaN/Inf in the data itself, out-of-range labels → check the data pipeline.
    - Learning rate too high / exploding gradients → lower lr, add warmup, clip gradients.
    - Numerically unstable code: `log(0)`, gradient of `sqrt(0)`, division by 0, `exp` overflow → use `log_softmax`, `BCEWithLogitsLoss`, add $\epsilon$, `clamp`.
    - fp16 overflow (> 65504) → use GradScaler, switch to bf16, keep attention/softmax/normalization layers in fp32.
    - An attention row fully masked to `-inf` makes softmax return NaN.
- With GradScaler, an occasional inf is normal (that step is skipped); persistent NaN is the real problem.

Q19: ***What are the common GPU/CPU memory leaks in PyTorch?***

- **Storing loss tensors in a list**: `losses.append(loss)` keeps **every step's entire computational graph** alive, so memory keeps growing. Use `losses.append(loss.item())` or `loss.detach()`.
- Running the validation loop without `torch.no_grad()`, so all intermediate activations are saved.
- Holding graph-attached tensors in globals, closures or hooks (e.g. a hook storing activations without `detach`).
- Overusing `retain_graph=True`.
- `.item()` / `.cpu()` force a GPU sync; calling them frequently hurts speed (not a leak, but a common performance pitfall).
- Tools: `torch.cuda.memory_allocated()` / `max_memory_allocated()`, `torch.cuda.memory_summary()`, memory snapshots via `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` and then visualize.
- Note: the memory shown by `nvidia-smi` includes blocks reserved by PyTorch's **caching allocator**; `empty_cache()` only returns cached memory to the driver and does not fix a real leak.

Q20: ***How do you profile a PyTorch program?***

- Coarse to fine: first check whether the GPU is saturated (`nvidia-smi`) — if utilization is low, the bottleneck is usually **data loading** (increase `num_workers`, `pin_memory`, move preprocessing offline) or CPU-side overhead/frequent syncs.
- **torch.profiler**: records CPU/CUDA op timings, memory and call stacks, and exports Chrome traces / TensorBoard timelines.

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True, profile_memory=True,
) as prof:
    for step, (x, y) in enumerate(loader):
        train_step(x, y)
        prof.step()
        if step >= 5:
            break
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

- Pitfalls: CUDA is **asynchronous**, so call `torch.cuda.synchronize()` before and after timing with `time.time()` (or use `torch.cuda.Event`); skip warmup (cuDNN tuning, compilation); hidden sync points (`.item()`, `print(tensor)`, `nonzero`, boolean indexing) stall the pipeline.
- Optimization directions: mixed precision, `torch.compile`, larger batches, `channels_last`, FlashAttention (`F.scaled_dot_product_attention`), fewer host-to-device copies and syncs.

Q21: ***What hooks does PyTorch provide and what are they used for?***

- **Module hooks**:
    - `register_forward_pre_hook(fn(module, args))`: before forward; can modify inputs.
    - `register_forward_hook(fn(module, args, output))`: after forward; can read/replace outputs, commonly used to **extract intermediate features**, visualize, or check for NaN.
    - `register_full_backward_hook(fn(module, grad_input, grad_output))`: gets gradients w.r.t. the module's inputs/outputs during backward (e.g. Grad-CAM).
- **Tensor hooks**: `t.register_hook(fn(grad))`, called after the tensor's gradient is computed; can modify the gradient (gradient clipping, gradient reversal, recording gradients of non-leaf tensors).
- There is also `register_post_accumulate_grad_hook` (called after a parameter's gradient has been accumulated; can run the optimizer step per parameter during backward to save memory); DDP/FSDP also implement gradient communication with hooks internally.

```python
feats = {}
def save_output(name):
    def hook(module, args, output):
        feats[name] = output.detach()      # detach to avoid holding the graph
    return hook

h = model.layer3.register_forward_hook(save_output("layer3"))
model(x)
h.remove()                                 # remove when done, otherwise it stays active
```

- Pitfalls: storing non-detached tensors in hooks leaks memory; forgetting `remove()`; Python code with side effects inside hooks can cause graph breaks or recompilation under `torch.compile`.
