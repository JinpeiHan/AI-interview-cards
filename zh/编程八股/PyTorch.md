# PyTorch

## 张量基础

Q1：***PyTorch Tensor 和 NumPy ndarray 有什么区别？如何互相转换？***

- **设备**：Tensor 可以放在 GPU/MPS 等加速器上（`.to("cuda")`），ndarray 只在 CPU。
- **自动求导**：Tensor 可以 `requires_grad=True` 并记录计算图，ndarray 不能。
- **默认 dtype**：`torch.tensor([1.0])` 默认 `float32`，`np.array([1.0])` 默认 `float64`——从 NumPy 转过来的数据常常意外变成 float64，导致与模型权重 dtype 不匹配或变慢。
- 转换：
    - `torch.from_numpy(a)` 与 `t.numpy()`：**共享内存**（零拷贝），修改一方另一方也变；仅限 CPU 张量。
    - `torch.tensor(a)`：**总是拷贝**。
    - GPU 或需要梯度的张量要先 `t.detach().cpu().numpy()`。
- API 风格接近（广播、索引、`sum(dim=...)` 对应 `axis=...`），但 PyTorch 还有 in-place 版本（带下划线 `add_`）。

Q2：***view、reshape、transpose、contiguous 有什么区别？什么时候是视图，什么时候是拷贝？***

- 张量 = 底层**存储（storage）** + `shape` + `stride` + `offset`。**视图**共享存储，只改元信息。
- **视图操作**：`view`、`transpose`/`t`、`permute`、切片 `x[:, 1:]`、`expand`、`squeeze/unsqueeze`、`narrow`、`detach`。
- `transpose/permute` 后 stride 不再是行主序，张量**不连续**（`is_contiguous()==False`）；此时 `view` 会报错，需要 `.contiguous().view(...)`（会拷贝）或直接用 `reshape`。
- `reshape`：能返回视图就返回视图，否则拷贝——所以**不能依赖它是否共享内存**。
- **拷贝操作**：`clone()`、`contiguous()`（不连续时）、高级索引 `x[idx_tensor]`、布尔索引 `x[mask]`、`torch.tensor(...)`。
- `expand` 不分配内存（stride 为 0），但对其做 in-place 写入是错误的；需要真正复制时用 `repeat`。

```python
a = torch.arange(6).view(2, 3)
b = a.t()                    # 视图，共享存储
b.is_contiguous()            # False
# b.view(-1)                 # RuntimeError
b.reshape(-1)                # 内部拷贝：tensor([0, 3, 1, 4, 2, 5])
a.data_ptr() == b.data_ptr() # True
```

Q3：***PyTorch 的广播规则是什么？有哪些常见的坑？***

- 从**最后一维**向前逐维对齐；每一维要么相等，要么其中一个为 1，要么其中一个不存在（视为 1）；结果取较大者。
- 例：`(B, T, C) + (C,)` → OK；`(B, 1, T) * (B, T, 1)` → `(B, T, T)`。
- 常见坑：
    - `y_pred` 形状 `(N, 1)`、`y` 形状 `(N,)`，`(y_pred - y)` 广播成 `(N, N)`，MSE 静默算错——`nn.MSELoss` 会对此给出警告。
    - In-place 操作不能改变左侧形状：`a.add_(b)` 要求 `b` 能广播到 `a`。
    - 广播出来的大张量可能悄悄占满显存（例如两两距离 `(N,1,D)-(1,M,D)`）。
- 习惯：关键位置加 `assert x.shape == (...)`，或用 `keepdim=True` 保持维度。

## 自动求导

Q4：***讲讲 autograd 和计算图的工作原理***

- PyTorch 是**动态图**（define-by-run）：前向执行时，每个对 `requires_grad` 张量的运算都会创建一个 `grad_fn` 节点（如 `MulBackward0`），并保存反向所需的张量（`ctx.save_for_backward`），形成一张 DAG。
- `loss.backward()` 从 loss 出发按**反向拓扑序**做**反向模式自动微分**（vector-Jacobian product），把梯度累加到叶子张量的 `.grad`。
- 反向完成后默认**释放图**及中间缓存；若需要再次反向，传 `retain_graph=True`（如多个 loss 共享一部分图且分开 backward）。
- 非标量输出调用 `backward` 需要传入 `gradient`（即 VJP 中的向量 $v$），例如 `y.backward(torch.ones_like(y))`。
- `torch.autograd.grad(outputs, inputs, create_graph=True)` 可以直接返回梯度而不写入 `.grad`，`create_graph=True` 用于高阶导（如梯度惩罚、MAML）。
- 因为动态图，Python 控制流（if / for / 递归）可直接参与建图，每次迭代图都可以不同。

Q5：***requires_grad、torch.no_grad、torch.inference_mode、detach 分别是什么？有什么区别？***

- `requires_grad=True`：标记张量需要梯度；对它的运算会被记录。`nn.Parameter` 默认为 True。冻结参数：`p.requires_grad_(False)`。
- `torch.no_grad()`：上下文管理器，块内运算**不建图**，输出 `requires_grad=False`；用于验证、EMA 更新、手动修改权重。
- `torch.inference_mode()`：比 `no_grad` 更激进，还关闭了视图追踪和版本计数，速度更快；但它产生的 **inference tensor 之后不能再参与 autograd**（不能用于需要梯度的计算）。纯推理/部署首选。
- `x.detach()`：返回**共享存储**、但脱离计算图的新张量（`requires_grad=False`）；用于截断梯度（如 target network、stop-gradient、记录 loss 数值）。注意修改 detach 后的张量会影响原张量。
- `torch.set_grad_enabled(mode)`：按条件开关，常用于同一个函数同时服务 train/val。
- 与 `model.eval()` 是**正交**的：`eval()` 改变 Dropout/BN 行为，不关闭梯度；推理时两者都要用。

Q6：***什么是叶子张量？为什么梯度会累加？为什么每步要 zero_grad？***

- **叶子张量**：用户直接创建的张量（`requires_grad=True`），或 `requires_grad=False` 的张量；它们的 `grad_fn is None`。只有叶子张量在 `backward` 后默认会填充 `.grad`；非叶子张量要 `.retain_grad()` 才会保留。
- `.grad` 是**累加**（`+=`）而不是覆盖：这样设计是为了支持一个参数在图中被多次使用、多个 loss 分别 backward、以及梯度累积。

```python
w = torch.ones(3, requires_grad=True)
for _ in range(2):
    (w * 2).sum().backward()
print(w.grad)   # tensor([4., 4., 4.])，两次累加
```

- 因此每次优化步之前要清零：`optimizer.zero_grad(set_to_none=True)`（PyTorch 2.x 的默认值），把 `.grad` 置为 `None` 而不是填 0，省内存且更快；但之后读取 `.grad` 要注意可能是 `None`。
- 坑：对 `.cuda()` 或 `.to()` 之后的张量求梯度——`w = torch.randn(3, requires_grad=True).cuda()` 得到的是**非叶子**，`w.grad` 为 None。应该 `torch.randn(3, device="cuda", requires_grad=True)`。

Q7：***如何实现自定义 autograd Function？什么场景下需要？***

- 继承 `torch.autograd.Function`，实现静态方法 `forward(ctx, ...)` 和 `backward(ctx, grad_out)`，用 `ctx.save_for_backward` 保存张量；通过 `.apply` 调用。
- 场景：数值更稳定或更省显存的梯度公式、调用自定义 CUDA/Triton kernel、straight-through estimator（量化）、梯度反转层（GRL）。

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
        return grad_out * (s + x * s * (1 - s))   # 每个 forward 输入对应一个梯度

x = torch.randn(5, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(Swish.apply, (x,))   # True：用有限差分验证 backward
```

- 要点：`backward` 返回的梯度个数等于 `forward` 的输入个数，非张量输入返回 `None`；用 `gradcheck`（double 精度）验证；需要 `torch.compile` 兼容或高阶导时要额外注意（如 `setup_context`、`once_differentiable`）。

Q8：***In-place 操作会对 autograd 造成什么问题？***

- 反向需要用到的张量如果被原地修改，会触发 `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`。autograd 通过张量的**版本计数器**（`_version`）检测这一点。
- 例：`y = x.sigmoid(); y.mul_(2)`——`sigmoid` 的反向需要输出 `y`，被改写后就出错。
- 叶子张量 `requires_grad=True` 时不能直接 in-place 修改（需在 `no_grad` 下，如优化器更新参数）。
- 建议：除非确实有显存压力，否则少用 in-place（`nn.ReLU(inplace=True)` 一般安全，因为 ReLU 反向只用输出的符号）。

## 模块与数据

Q9：***nn.Module 中的 parameters、buffers、子模块分别是什么？***

- `nn.Module` 通过 `__setattr__` 自动注册：
    - **Parameter**（`nn.Parameter`）：可学习参数，出现在 `parameters()` 中，会被优化器更新。
    - **Buffer**（`register_buffer`）：非学习的状态，如 BN 的 `running_mean`、RoPE 的 cos/sin 缓存、因果 mask；会随 `.to(device)` 移动，默认存进 `state_dict`；`persistent=False` 则不保存。
    - **子模块**：赋值为属性的其他 `nn.Module`，递归管理。
- 坑：
    - 用 Python `list`/`dict` 装子模块不会被注册（参数不在 `parameters()` 里、不随 `.to()` 移动），要用 `nn.ModuleList` / `nn.ModuleDict` / `nn.ParameterList`。
    - 普通张量属性 `self.mask = torch.ones(...)` 不会随 `.to("cuda")` 移动，应 `register_buffer`。
- 常用 API：`named_parameters()`、`modules()`、`apply(fn)`（如自定义初始化）、`state_dict()`、`requires_grad_()`、`zero_grad()`。

Q10：***model.train() 和 model.eval() 有什么区别？***

- 只是递归地设置每个子模块的 `self.training` 标志，影响**依赖模式的层**：
    - **Dropout**：train 随机丢弃并缩放 $1/(1-p)$；eval 恒等。
    - **BatchNorm**：train 用 batch 统计并更新 running stats；eval 用 running stats。
- 它**不会**关闭梯度计算，推理时需要再配合 `torch.no_grad()` / `inference_mode()`。
- 常见 bug：验证完忘记切回 `model.train()`；在 eval 模式下微调 BN；batch 很小时训练模式的 BN 不稳定，可冻结 BN（对 BN 子模块调用 `eval()` 并冻结其参数）。

Q11：***Dataset 与 DataLoader 的工作机制？num_workers、pin_memory、collate_fn、sampler 的作用？***

- **Dataset**：map-style 实现 `__len__` 和 `__getitem__(i)`；iterable-style（`IterableDataset`）实现 `__iter__`，适合流式数据。
- **DataLoader** 负责批处理、打乱、并行加载：
    - `sampler` / `batch_sampler`：决定索引顺序。`shuffle=True` 就是 `RandomSampler`；类别不平衡用 `WeightedRandomSampler`；DDP 用 `DistributedSampler`（每个 epoch 调 `sampler.set_epoch(epoch)`，否则每轮打乱顺序相同）。
    - `collate_fn`：把样本列表合成 batch；变长序列在这里做 padding 并生成 mask。
    - `num_workers`：子进程数并行执行 `__getitem__`；0 表示在主进程加载。配合 `persistent_workers=True`（避免每个 epoch 重建进程）和 `prefetch_factor`。
    - `pin_memory=True`：把 batch 放到**页锁定内存**，使 H2D 拷贝更快，并可配合 `.to("cuda", non_blocking=True)` 与计算重叠。
    - `drop_last`：丢弃最后不完整的 batch（BN 或需要固定形状时有用）。
- 坑：多 worker 下每个 worker 的随机数种子要用 `worker_init_fn` / `generator` 控制（否则 NumPy 增强在不同 worker 中可能重复）；`IterableDataset` 要用 `get_worker_info()` 手动分片，否则数据会重复；`__getitem__` 里不要做 CUDA 操作。

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

## 训练技巧

Q12：***混合精度训练怎么做？autocast、GradScaler 的作用？fp16 和 bf16 有什么区别？***

- `torch.autocast`：在上下文中按算子自动选择精度——matmul/conv 用低精度，softmax、归约、loss 等数值敏感算子保持 fp32。参数（master weights）仍是 fp32。
- **fp16**：5 位指数，动态范围小，小梯度会**下溢为 0** → 需要 `GradScaler`：先把 loss 乘一个大系数再反向，`step` 前反缩放；若发现 inf/nan 就**跳过该步**并减小 scale。
- **bf16**：8 位指数（与 fp32 相同动态范围），尾数精度低；一般**不需要 GradScaler**。A100/H100 等新硬件上的首选。

```python
scaler = torch.amp.GradScaler("cuda")          # bf16 时可以不用
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                 # 裁剪前必须先反缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                     # 有 inf/nan 时跳过
    scaler.update()
```

- 坑：`backward` 不要放在 autocast 内（不需要）；梯度裁剪前忘记 `unscale_`；在 autocast 外自己算 loss 时注意 dtype 混用。

Q13：***如何实现梯度累积和梯度裁剪？***

- **梯度累积**：显存不够大 batch 时，连续 $k$ 个 micro-batch 只 backward 不 step，利用 `.grad` 的累加特性，等效 batch 为 $k\times$。loss 要**除以 $k$** 才与大 batch 的均值等价。
- **梯度裁剪**：`clip_grad_norm_` 把所有参数梯度的全局 L2 范数裁到 `max_norm` 以内（$g\leftarrow g\cdot\min(1,\frac{c}{\|g\|})$），防止梯度爆炸，Transformer 常用 1.0；它会返回裁剪前的范数，可用于监控。`clip_grad_value_` 逐元素裁剪，较少用。

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

- 坑：BN 的统计量仍按 micro-batch 计算，不等价于大 batch；DDP 下非同步步要用 `with model.no_sync():` 避免每个 micro-batch 都做 all-reduce；裁剪要在**累积完成后、step 之前**做；按 token 数加权的 loss（变长序列）应按总 token 数归一化而不是简单除以 $k$。

Q14：***DataParallel 和 DistributedDataParallel（DDP）有什么区别？***

- **DataParallel（DP）**：单进程多线程；每步把模型从主卡**复制**到各卡、scatter 输入、gather 输出到主卡计算 loss，再 reduce 梯度到主卡。受 **GIL** 限制，主卡显存和通信成为瓶颈，只支持单机。已不推荐。
- **DDP**：**每个 GPU 一个进程**，每个进程持有完整模型副本，初始化时广播参数；反向时通过 autograd hook 把梯度分**桶（bucket）**做 **all-reduce**（Ring/NCCL），并与反向计算**重叠**；每个进程各自 `optimizer.step()`，参数保持一致。支持多机。
- 用法要点：`torchrun --nproc_per_node=N train.py`；`dist.init_process_group("nccl")`；`DistributedSampler` + `set_epoch`；只在 rank 0 保存/打印；保存时用 `model.module.state_dict()`。
- 有些参数在前向中没被使用时要设 `find_unused_parameters=True`（有额外开销）。
- 追问：模型放不下单卡 → **FSDP**（PyTorch 2.x 推荐 `fully_shard`/FSDP2，切分参数、梯度、优化器状态，对应 ZeRO-3）、张量并行、流水线并行。

Q15：***torch.compile 是什么？有什么好处和注意事项？***

- PyTorch 2.0 引入：`model = torch.compile(model)`。**TorchDynamo** 在 Python 字节码层面捕获 FX 图，**AOTAutograd** 同时生成反向图，**Inductor** 后端做算子融合并生成 Triton（GPU）/C++（CPU）kernel。
- 好处：减少 kernel 启动和显存读写（如把 elementwise 链融合），无需改代码即可提速；`mode="reduce-overhead"` 使用 CUDA Graphs 降低小 batch 开销，`mode="max-autotune"` 做更多调优。
- 注意：
    - 首次调用会编译，耗时较长；**输入形状变化**会触发重编译（可设 `dynamic=True` 或对齐/分桶形状）。
    - **Graph break**：遇到不支持的 Python 代码（打印、`.item()`、数据相关控制流、某些第三方库）会把图切断，回退到 eager，降低收益；可用 `TORCH_LOGS="graph_breaks"` 或 `torch._dynamo.explain` 排查。
    - 结果与 eager 存在浮点级差异；调试时先在 eager 下确认正确。

Q16：***如何保存和加载模型？为什么推荐 state_dict？***

- **推荐**：保存 `state_dict`（参数和 buffer 的有序字典），与代码解耦，可移植。
- `torch.save(model)` 保存整个对象依赖 pickle 和类定义路径，代码重构后容易加载失败，也有安全风险。PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，只允许加载张量等安全类型。
- 断点续训要同时保存优化器、调度器、`GradScaler`、epoch/step 和随机数状态。

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
}
torch.save(ckpt, "ckpt.pt")

ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])            # strict=True 时键必须完全匹配
optimizer.load_state_dict(ckpt["optimizer"])
```

- 坑：DDP / `torch.compile` 包装后的键名带 `module.` / `_orig_mod.` 前缀，保存原始模块或加载时去掉前缀；用 `map_location` 避免在错误的 GPU 上加载；先 `model.to(device)` 再创建优化器（或加载优化器状态）；部分加载（迁移学习）用 `strict=False` 并检查返回的 `missing_keys` / `unexpected_keys`。大模型常用 `safetensors` 格式（安全、支持 mmap 快速加载）。

Q17：***如何保证实验可复现？***

- 固定所有随机种子：

```python
import random, numpy as np, torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)              # 同时设置 CPU 和所有 CUDA 设备
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
torch.backends.cudnn.benchmark = False              # 不自动选择（可能不同的）卷积算法
torch.use_deterministic_algorithms(True)            # 对非确定性算子报错
# 部分 cuBLAS 操作还需要环境变量 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

- DataLoader：传入固定种子的 `generator`，并用 `worker_init_fn` 给每个 worker 设种子。
- 非确定性来源：cuDNN 自动调优、`atomicAdd` 类的 scatter/index_add 反向、多 GPU 通信的归约顺序、不同硬件/驱动/库版本。
- 实话实说：**bitwise 复现**只能在相同硬件和软件栈上实现，且会牺牲速度；通常目标是统计上可复现（多种子取均值和方差）。

## 调试与性能

Q18：***训练中 loss 出现 NaN/Inf，如何排查？***

- **定位**：
    - `torch.autograd.set_detect_anomaly(True)`：反向出现 NaN 时报出对应的前向算子（很慢，仅调试用）。
    - 在前向中加 `assert torch.isfinite(x).all()`，或注册 forward hook 检查每层输出。
    - 记录梯度范数（`clip_grad_norm_` 的返回值），观察爆炸发生在哪一步；检查出问题的那个 batch 数据。
- **常见原因与修复**：
    - 数据本身有 NaN/Inf、标签越界 → 检查数据管道。
    - 学习率过大 / 梯度爆炸 → 降 lr、加 warmup、梯度裁剪。
    - 数值不稳定的写法：`log(0)`、`sqrt(0)` 的梯度、除以 0、`exp` 溢出 → 用 `log_softmax`、`BCEWithLogitsLoss`、加 $\epsilon$、`clamp`。
    - fp16 溢出（> 65504）→ 用 GradScaler、改用 bf16、对 attention/softmax/归一化层保持 fp32。
    - 注意力中整行被 mask 成 `-inf` 导致 softmax 得到 NaN。
- 用 GradScaler 时偶发的 inf 是正常的（该步会被跳过）；持续 NaN 才是问题。

Q19：***PyTorch 中常见的显存/内存泄漏有哪些？***

- **把 loss 张量存进列表**：`losses.append(loss)` 会让每一步的**整张计算图**都无法释放，显存持续增长。应 `losses.append(loss.item())` 或 `loss.detach()`。
- 验证循环没有用 `torch.no_grad()`，中间激活全部被保存。
- 在全局变量、闭包或 hook 中持有带图的张量（如 hook 中保存了未 `detach` 的激活）。
- `retain_graph=True` 被滥用。
- `.item()` / `.cpu()` 会触发 GPU 同步，频繁调用影响速度（不是泄漏，但是常见的性能坑）。
- 排查工具：`torch.cuda.memory_allocated()` / `max_memory_allocated()`、`torch.cuda.memory_summary()`、显存快照 `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()` 后可视化。
- 说明：`nvidia-smi` 看到的显存包含 PyTorch **缓存分配器**保留的块，`empty_cache()` 只会把缓存还给驱动，并不能解决真正的泄漏。

Q20：***如何对 PyTorch 程序做性能分析（profiling）？***

- 先粗后细：先看 GPU 利用率是否满载（`nvidia-smi`）——若很低，通常是**数据加载**瓶颈（增大 `num_workers`、`pin_memory`、预处理离线化）或 CPU 端开销/频繁同步。
- **torch.profiler**：记录 CPU/CUDA 算子耗时、显存、调用栈，可导出 Chrome trace / TensorBoard 查看时间线。

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

- 坑：CUDA 是**异步**的，用 `time.time()` 计时前后要 `torch.cuda.synchronize()`（或用 `torch.cuda.Event`）；要跳过 warmup（cuDNN 调优、编译）；隐藏的同步点（`.item()`、`print(tensor)`、`nonzero`、布尔索引）会打断流水线。
- 优化方向：混合精度、`torch.compile`、更大 batch、`channels_last`、FlashAttention（`F.scaled_dot_product_attention`）、减少 H2D 拷贝和同步。

Q21：***PyTorch 的 hook 有哪些？分别用来做什么？***

- **Module hook**：
    - `register_forward_pre_hook(fn(module, args))`：前向之前，可修改输入。
    - `register_forward_hook(fn(module, args, output))`：前向之后，可读取/替换输出，常用于**提取中间特征**、可视化、检查 NaN。
    - `register_full_backward_hook(fn(module, grad_input, grad_output))`：反向时获取模块输入/输出的梯度（如 Grad-CAM）。
- **Tensor hook**：`t.register_hook(fn(grad))`，在该张量梯度计算后调用，可修改梯度（如梯度裁剪、梯度反转、记录非叶子张量的梯度）。
- 还有 `register_post_accumulate_grad_hook`（参数梯度累加完后调用，可用于在反向过程中逐参数执行优化器步以省显存），DDP/FSDP 内部也用 hook 实现梯度通信。

```python
feats = {}
def save_output(name):
    def hook(module, args, output):
        feats[name] = output.detach()      # detach，避免持有计算图
    return hook

h = model.layer3.register_forward_hook(save_output("layer3"))
model(x)
h.remove()                                 # 用完要移除，否则一直生效
```

- 坑：hook 中保存未 detach 的张量会导致显存泄漏；忘记 `remove()`；hook 中有副作用的 Python 代码可能导致 graph break 或重编译，影响 `torch.compile`。
