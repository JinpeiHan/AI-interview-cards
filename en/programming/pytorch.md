# PyTorch

## Tensor Basics

Q1: ***What are the differences between a PyTorch Tensor and a NumPy ndarray? How do you convert between them?***

- **A Tensor can live on a GPU and supports autograd; an ndarray is CPU-only with no gradients; the APIs are very similar. `torch.from_numpy` / `.numpy()` share memory, while `torch.tensor` always copies.**
- Default dtypes differ: Tensor uses float32, NumPy float64, so converted data often mismatches the model weights.
- GPU tensors or tensors requiring grad need `t.detach().cpu().numpy()` first.

Q2: ***What is the difference between view, reshape, transpose and contiguous? When do you get a view and when a copy?***

- **A view only changes shape and strides and shares the storage. After `transpose` the tensor is non-contiguous and `view` raises an error, so call `contiguous()` first (copies) or use `reshape` (a view when possible, otherwise a copy).**
- Views: `view`, `transpose`/`permute`, slicing, `expand`, `squeeze/unsqueeze`, `detach`.
- Copies: `clone()`, advanced/boolean indexing, `torch.tensor(...)`; `expand` allocates no memory and must not be written in place, so use `repeat` for a real copy.

```python
a = torch.arange(6).view(2, 3)
b = a.t()                    # view, shares storage
b.is_contiguous()            # False
# b.view(-1)                 # RuntimeError
b.reshape(-1)                # copies internally: tensor([0, 3, 1, 4, 2, 5])
a.data_ptr() == b.data_ptr() # True
```

Q3: ***What are PyTorch's broadcasting rules? What are the common pitfalls?***

- **Align dimensions from the last one backwards; each dimension must be equal or have one side equal to 1 (a missing dimension counts as 1), and the result takes the larger size.**
- The most common pitfall: predictions (N, 1) minus labels (N,) broadcast to (N, N), so the loss is silently wrong.
- Broadcasting can silently create huge tensors that fill GPU memory; add shape assertions at key points or use `keepdim=True`.

## Autograd

Q4: ***Explain how autograd and the computational graph work***

- **PyTorch uses a dynamic graph: during the forward pass every op that needs gradients records a backward node and saves the tensors it needs; `backward()` runs reverse-mode automatic differentiation from the loss in reverse order and accumulates gradients into the `.grad` of leaf tensors.**
- After backward the graph is freed by default; backpropagating again needs `retain_graph=True`.
- `torch.autograd.grad(..., create_graph=True)` supports higher-order derivatives (e.g. gradient penalty).
- Because the graph is dynamic, Python if/for take part in graph construction directly, and the graph can differ every iteration.

Q5: ***What are requires_grad, torch.no_grad, torch.inference_mode and detach? How do they differ?***

- **`requires_grad` marks whether a tensor records gradients; `no_grad` makes a block of code build no graph; `inference_mode` is a faster, more thorough `no_grad`; `detach` returns a tensor that shares storage but is cut off from the graph.**
- Tensors produced under `inference_mode` can never take part in autograd later, so it suits pure inference; `detach` is used to stop gradients (target networks, stop-gradient).
- None of these relate to `model.eval()`: `eval()` only changes Dropout/BN behavior and does not disable gradients, so use both at inference.

Q6: ***What is a leaf tensor? Why do gradients accumulate? Why call zero_grad every step?***

- **Leaf tensors are tensors created directly by the user, with no `grad_fn`, and only they get `.grad` after backward by default. `.grad` accumulates, so it must be cleared before each update.**
- Accumulation supports a parameter used several times in the graph, several losses backpropagated separately, and gradient accumulation.
- Pitfall: `torch.randn(3, requires_grad=True).cuda()` is a non-leaf, so `.grad` is None; use `torch.randn(3, device="cuda", requires_grad=True)` instead.

```python
w = torch.ones(3, requires_grad=True)
for _ in range(2):
    (w * 2).sum().backward()
print(w.grad)   # tensor([4., 4., 4.]), accumulated twice
```

Q7: ***How do you implement a custom autograd Function? When is it needed?***

- **Subclass `torch.autograd.Function`, write the static methods `forward` and `backward` yourself, save what backward needs on `ctx`, and call it via `.apply`. You only need it when you want to control how the gradient is computed.**
- Use cases: more stable or memory-efficient gradients, custom CUDA/Triton kernels, straight-through estimators (STE) in quantization, gradient reversal layers.
- Key points: `backward` returns as many gradients as `forward` has inputs; verify with `gradcheck` (double precision).

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

Q8: ***What problems do in-place operations cause for autograd?***

- **If a tensor needed by backward has been modified in place, backward raises an error; autograd detects this through each tensor's version counter.**
- Example: `y = x.sigmoid(); y.mul_(2)` fails because sigmoid's backward needs y, which was changed.
- A leaf tensor that requires grad can only be modified in place under `no_grad` (as optimizers do); avoid in-place ops unless memory is tight.

## Modules and Data

Q9: ***What are parameters, buffers and submodules in an nn.Module?***

- **Parameters are learnable and updated by the optimizer; buffers are non-learned model state (e.g. BN's running_mean, a causal mask), moved by `.to(device)` and saved in `state_dict`; submodules are other Modules attached as attributes and managed recursively.**
- Pitfall: submodules held in a plain list/dict are not registered; use `nn.ModuleList` / `nn.ModuleDict`.
- Pitfall: a plain tensor attribute is not moved by `.to("cuda")`; use `register_buffer`.

Q10: ***What is the difference between model.train() and model.eval()?***

- **They only set each submodule's `training` flag, which affects Dropout (random drops in training, identity in eval) and BatchNorm (batch statistics in training, running stats in eval); they do not disable gradients.**
- At inference also use `no_grad` / `inference_mode`; a common bug is forgetting to switch back to `train()` after validation.

Q11: ***How do Dataset and DataLoader work? What do num_workers, pin_memory, collate_fn and sampler do?***

- **A Dataset "returns one sample for an index"; a DataLoader fetches samples in the order given by the sampler, merges them into batches with collate_fn, and loads in parallel with several worker processes.**
- `sampler` decides the order (for DDP use `DistributedSampler` and call `set_epoch` every epoch); `collate_fn` does variable-length padding and masks.
- `num_workers` is the number of loading processes; `pin_memory=True` uses page-locked memory so copies to the GPU are faster and can overlap with compute.
- Pitfalls: seed each worker separately; an `IterableDataset` must be sharded per worker manually, otherwise data is duplicated.

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

- **autocast runs matmuls and convolutions in low precision while keeping sensitive ops such as softmax, reductions and losses in fp32; parameters stay fp32. fp16 has a small dynamic range and small gradients underflow, so it needs GradScaler; bf16 has the same dynamic range as fp32, usually needs none, and is the first choice on newer hardware.**
- GradScaler: scale the loss up before backward and back down before the update; on inf/nan, skip that step and lower the scale.
- Pitfall: you must `unscale_` before gradient clipping.

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

Q13: ***How do you implement gradient accumulation and gradient clipping?***

- **Gradient accumulation: run backward on k small batches without updating, relying on `.grad` accumulating, for an effective k-times batch; divide the loss by k. Gradient clipping: cap the overall norm of all gradients at a threshold to prevent explosion; 1.0 is common for Transformers.**
- Clip after accumulation is complete and before `step`.
- Pitfalls: BN statistics are still per small batch; under DDP use `no_sync()` on intermediate steps to avoid syncing gradients every step.

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

Q14: ***What is the difference between DataParallel and DistributedDataParallel (DDP)?***

- **DP is single-process, multi-threaded: every step it copies the model from the main GPU and gathers outputs and gradients back to it, so it is GIL-bound, bottlenecked on the main GPU and single-machine only; it is no longer recommended. DDP runs one process per GPU, each with a full model copy, all-reduces gradients in buckets during backward while overlapping with compute, and supports multiple machines.**
- DDP essentials: launch with `torchrun`, `DistributedSampler` + `set_epoch`, save and print only on rank 0.
- Follow-up: when the model does not fit on one GPU, use FSDP (shards parameters, gradients and optimizer state, equivalent to ZeRO-3), tensor parallelism or pipeline parallelism.

Q15: ***What is torch.compile? What are its benefits and caveats?***

- **PyTorch 2.0's compiler: `torch.compile(model)` captures Python code as a graph, fuses ops and generates Triton/C++ kernels, giving speedups without code changes.**
- Components: TorchDynamo captures the graph, AOTAutograd builds the backward graph, Inductor generates kernels.
- Caveat: the first call compiles; changing input shapes triggers recompilation (`dynamic=True` helps).
- Caveat: prints, `.item()` and data-dependent branches cause graph breaks that fall back to eager and reduce the gains.

Q16: ***How do you save and load models? Why is state_dict recommended?***

- **Save only the `state_dict` (a dict of parameters and buffers): it is decoupled from code and portable. Saving the whole model relies on pickle and class paths, so it can break as soon as the code changes, and it is a security risk.**
- To resume training also save the optimizer, scheduler, step count and RNG states; since 2.6 `torch.load` defaults to `weights_only=True`.

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

- Pitfalls: keys of DDP / `torch.compile`-wrapped models carry `module.` / `_orig_mod.` prefixes; use `map_location` to avoid loading onto the wrong GPU; for partial loading use `strict=False` and check the missing keys.

Q17: ***How do you make experiments reproducible?***

- **Fix all random seeds (Python, NumPy, PyTorch), turn off cuDNN algorithm autotuning and enable deterministic algorithms, and seed the DataLoader workers too.**
- Sources of non-determinism: cuDNN autotuning, backward of atomic-add-style ops, reduction order across GPUs, hardware and library versions.
- Honest answer: bitwise reproducibility is only possible on identical hardware and software and costs speed; the usual goal is statistical reproducibility (mean and variance over several seeds).

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

## Debugging and Performance

Q18: ***How do you debug a NaN/Inf loss during training?***

- **First locate the step, layer and batch where it goes wrong, then check the usual causes: bad data, a learning rate that is too high, numerically unstable code, or fp16 overflow.**
- Locate: `set_detect_anomaly(True)` reports the forward op that produced the NaN (very slow); check each layer's output with hooks; log the gradient norm.
- Fix: clean the data; lower the learning rate, add warmup, clip gradients; use `log_softmax`, `BCEWithLogitsLoss`, add ε; for fp16 overflow switch to bf16.
- With GradScaler an occasional inf is normal (that step is skipped); only persistent NaN is a problem.

Q19: ***What are the common GPU/CPU memory leaks in PyTorch?***

- **The most common one is accidentally keeping the computational graph referenced: storing loss tensors directly in a list, or keeping non-detached tensors in globals or hooks, so no step's graph is freed and memory keeps growing.**
- Fixes: record values with `loss.item()`; wrap the validation loop in `no_grad()`; do not overuse `retain_graph=True`.
- Diagnose with `memory_allocated()`, `memory_summary()` and memory snapshots; `nvidia-smi` includes memory reserved by the caching allocator, and `empty_cache()` does not fix a real leak.

Q20: ***How do you profile a PyTorch program?***

- **Coarse to fine: first check GPU utilization; if it is low, the cause is usually data loading, CPU overhead or frequent syncs. Then use `torch.profiler` to see per-op time and memory and export a timeline to analyze.**
- Pitfalls: CUDA is asynchronous, so call `torch.cuda.synchronize()` before and after manual timing, and skip warmup; implicit syncs such as `.item()` or printing tensors stall the pipeline.
- Optimization directions: mixed precision, `torch.compile`, larger batches, FlashAttention, fewer copies and syncs.

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

Q21: ***What hooks does PyTorch provide and what are they used for?***

- **Two kinds: module hooks attach before forward, after forward or during backward, to modify inputs, extract intermediate features or inspect gradients (e.g. Grad-CAM); tensor hooks are called once that tensor's gradient is computed and can read or modify it (e.g. gradient reversal).**
- DDP/FSDP also rely on hooks internally for gradient communication.
- Pitfalls: storing non-detached tensors in a hook leaks memory; call `remove()` when done.

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
