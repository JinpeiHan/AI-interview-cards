# PyTorch

## 张量基础

Q1：***PyTorch Tensor 和 NumPy ndarray 有什么区别？如何互相转换？***

- **Tensor 能放 GPU、能自动求导，ndarray 只在 CPU、不能求导，API 很像。`torch.from_numpy` / `.numpy()` 共享内存，`torch.tensor` 总是拷贝。**
- 默认 dtype 不同：Tensor 是 float32，NumPy 是 float64，转过来常和模型权重不匹配。
- GPU 或需要梯度的张量要先 `t.detach().cpu().numpy()`。

Q2：***view、reshape、transpose、contiguous 有什么区别？什么时候是视图，什么时候是拷贝？***

- **视图只改形状和步长、共享存储。`transpose` 后张量不连续，`view` 会报错，要先 `contiguous()`（拷贝）或用 `reshape`（能给视图就给，否则拷贝）。**
- 视图：`view`、`transpose`/`permute`、切片、`expand`、`squeeze/unsqueeze`、`detach`。
- 拷贝：`clone()`、高级/布尔索引、`torch.tensor(...)`；`expand` 不分配内存，不能原地写，要真复制用 `repeat`。

```python
a = torch.arange(6).view(2, 3)
b = a.t()                    # 视图，共享存储
b.is_contiguous()            # False
# b.view(-1)                 # RuntimeError
b.reshape(-1)                # 内部拷贝：tensor([0, 3, 1, 4, 2, 5])
a.data_ptr() == b.data_ptr() # True
```

Q3：***PyTorch 的广播规则是什么？有哪些常见的坑？***

- **从最后一维往前对齐，每一维要么相等、要么有一方是 1（缺失当作 1），结果取较大者。**
- 最常见的坑：预测 (N, 1) 减标签 (N,) 会广播成 (N, N)，loss 静默算错。
- 广播可能悄悄生成大张量占满显存；关键处加形状断言或用 `keepdim=True`。

## 自动求导

Q4：***讲讲 autograd 和计算图的工作原理***

- **PyTorch 是动态图：前向时每个需要梯度的运算都记录一个反向节点并保存所需张量；`backward()` 从 loss 出发按反向顺序做反向模式自动微分，把梯度累加到叶子张量的 `.grad`。**
- 反向后图默认被释放，再反向一次需要 `retain_graph=True`。
- `torch.autograd.grad(..., create_graph=True)` 可做高阶导（如梯度惩罚）。
- 因为是动态图，Python 的 if/for 能直接参与建图，每次迭代图都可以不同。

Q5：***requires_grad、torch.no_grad、torch.inference_mode、detach 分别是什么？有什么区别？***

- **`requires_grad` 标记张量是否记录梯度；`no_grad` 让一段代码不建图；`inference_mode` 是更快更彻底的 `no_grad`；`detach` 返回共享存储但脱离计算图的张量。**
- `inference_mode` 产生的张量之后不能再参与求导，适合纯推理；`detach` 用于截断梯度（target network、stop-gradient）。
- 都与 `model.eval()` 无关：`eval()` 只改 Dropout/BN 行为，不关梯度，推理时两者都要用。

Q6：***什么是叶子张量？为什么梯度会累加？为什么每步要 zero_grad？***

- **叶子张量是用户直接创建、没有 `grad_fn` 的张量，只有它们反向后默认得到 `.grad`。`.grad` 是累加的，所以每步更新前要清零。**
- 累加是为了支持参数在图中被多次使用、多个 loss 分别反向和梯度累积。
- 坑：`torch.randn(3, requires_grad=True).cuda()` 是非叶子，`.grad` 为 None；应直接 `torch.randn(3, device="cuda", requires_grad=True)`。

```python
w = torch.ones(3, requires_grad=True)
for _ in range(2):
    (w * 2).sum().backward()
print(w.grad)   # tensor([4., 4., 4.])，两次累加
```

Q7：***如何实现自定义 autograd Function？什么场景下需要？***

- **继承 `torch.autograd.Function`，自己写静态方法 `forward` 和 `backward`，用 `ctx` 保存反向要用的张量，通过 `.apply` 调用。想自己控制梯度怎么算时才需要。**
- 场景：更稳定或更省显存的梯度、自定义 CUDA/Triton kernel、量化的直通估计（STE）、梯度反转层。
- 要点：`backward` 返回的梯度个数等于 `forward` 的输入个数；用 `gradcheck`（double 精度）验证。

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

Q8：***In-place 操作会对 autograd 造成什么问题？***

- **反向需要的张量如果被原地修改，反向时会报错；autograd 靠每个张量的版本计数器发现这一点。**
- 例：`y = x.sigmoid(); y.mul_(2)`，sigmoid 的反向需要 y，但它被改了。
- 需要梯度的叶子张量只能在 `no_grad` 下原地修改（优化器就是这样）；除非显存紧张，少用原地操作。

## 模块与数据

Q9：***nn.Module 中的 parameters、buffers、子模块分别是什么？***

- **Parameter 是可学习参数，由优化器更新；buffer 是不学习的模型状态（如 BN 的 running_mean、因果 mask），会随 `.to(device)` 移动并存进 `state_dict`；子模块是挂为属性的其他 Module，被递归管理。**
- 坑：普通 list/dict 装的子模块不会被注册，要用 `nn.ModuleList` / `nn.ModuleDict`。
- 坑：普通张量属性不会随 `.to("cuda")` 移动，要用 `register_buffer`。

Q10：***model.train() 和 model.eval() 有什么区别？***

- **它们只是设置每个子模块的 `training` 标志，影响 Dropout（训练时随机丢弃，eval 时恒等）和 BatchNorm（训练时用 batch 统计，eval 时用 running stats）；不会关闭梯度。**
- 推理时还要配合 `no_grad` / `inference_mode`；常见 bug 是验证完忘记切回 `train()`。

Q11：***Dataset 与 DataLoader 的工作机制？num_workers、pin_memory、collate_fn、sampler 的作用？***

- **Dataset 负责"给索引返回一个样本"；DataLoader 按 sampler 给的顺序取样本，用 collate_fn 拼成 batch，用多个 worker 进程并行加载。**
- `sampler` 决定顺序（DDP 用 `DistributedSampler`，每个 epoch 要 `set_epoch`）；`collate_fn` 里做变长 padding 和 mask。
- `num_workers` 是加载进程数；`pin_memory=True` 用页锁定内存，让拷到 GPU 更快且能与计算重叠。
- 坑：每个 worker 的随机种子要单独设；`IterableDataset` 要按 worker 手动分片，否则数据重复。

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

- **autocast 让矩阵乘、卷积用低精度，softmax、归约、loss 等敏感算子保持 fp32，参数仍是 fp32。fp16 动态范围小、小梯度会下溢，需要 GradScaler；bf16 动态范围和 fp32 相同，一般不需要，新硬件上首选。**
- GradScaler：先放大 loss 再反向，更新前缩回；遇到 inf/nan 就跳过这一步并减小倍数。
- 坑：梯度裁剪前必须先 `unscale_`。

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

Q13：***如何实现梯度累积和梯度裁剪？***

- **梯度累积：连续 k 个小 batch 只反向不更新，利用 `.grad` 累加等效 k 倍 batch，loss 要除以 k。梯度裁剪：把所有梯度的整体范数限制在阈值内防止爆炸，Transformer 常用 1.0。**
- 裁剪在累积完成后、`step` 之前做。
- 坑：BN 统计仍按小 batch 算；DDP 下中间步用 `no_sync()` 避免每步同步梯度。

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

Q14：***DataParallel 和 DistributedDataParallel（DDP）有什么区别？***

- **DP 是单进程多线程，每步从主卡复制模型、把输出和梯度汇总到主卡，受 GIL 限制、主卡成瓶颈、只能单机，已不推荐。DDP 每张卡一个进程，各自持有完整模型，反向时分桶 all-reduce 梯度并与计算重叠，支持多机。**
- DDP 要点：`torchrun` 启动、`DistributedSampler` + `set_epoch`、只在 rank 0 保存和打印。
- 追问：单卡放不下模型时用 FSDP（切分参数、梯度和优化器状态，相当于 ZeRO-3）、张量并行或流水线并行。

Q15：***torch.compile 是什么？有什么好处和注意事项？***

- **PyTorch 2.0 的编译器：`torch.compile(model)` 把 Python 代码捕获成图，做算子融合并生成 Triton/C++ kernel，不改代码就能提速。**
- 组件：TorchDynamo 捕获图，AOTAutograd 生成反向图，Inductor 生成 kernel。
- 注意：首次调用要编译；输入形状变化会重编译（可用 `dynamic=True`）。
- 注意：打印、`.item()`、依赖数据的分支会造成 graph break，回退 eager、收益下降。

Q16：***如何保存和加载模型？为什么推荐 state_dict？***

- **推荐只保存 `state_dict`（参数和 buffer 的字典），它和代码解耦、可移植；保存整个模型依赖 pickle 和类路径，代码一改就可能加载失败，还有安全风险。**
- 断点续训还要保存优化器、调度器、步数和随机数状态；2.6 起 `torch.load` 默认 `weights_only=True`。

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

- 坑：DDP / `torch.compile` 包装后的键名带 `module.` / `_orig_mod.` 前缀；用 `map_location` 避免加载到错误的 GPU；部分加载用 `strict=False` 并检查缺失的键。

Q17：***如何保证实验可复现？***

- **固定所有随机种子（Python、NumPy、PyTorch），关闭 cuDNN 自动选算法并开启确定性算法，DataLoader 的 worker 也要设种子。**
- 非确定性来源：cuDNN 调优、原子加类算子的反向、多卡归约顺序、硬件和库版本。
- 实话实说：逐比特复现只在相同软硬件上可行且会变慢，通常追求统计上可复现（多种子取均值方差）。

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

## 调试与性能

Q18：***训练中 loss 出现 NaN/Inf，如何排查？***

- **先定位出问题的步、层和 batch，再查常见原因：数据有问题、学习率太大、数值不稳定的写法、fp16 溢出。**
- 定位：`set_detect_anomaly(True)` 报出产生 NaN 的前向算子（很慢）；用 hook 检查每层输出；记录梯度范数。
- 修复：清洗数据；降学习率、加 warmup、梯度裁剪；用 `log_softmax`、`BCEWithLogitsLoss`、加 ε；fp16 溢出改 bf16。
- 用 GradScaler 时偶发 inf 是正常的（该步被跳过），持续 NaN 才是问题。

Q19：***PyTorch 中常见的显存/内存泄漏有哪些？***

- **最常见的是无意中一直引用着计算图：把 loss 张量直接存进列表，或在全局变量、hook 里保存未 detach 的张量，每步的图都释放不掉，显存持续增长。**
- 修复：记录数值用 `loss.item()`；验证循环加 `no_grad()`；不滥用 `retain_graph=True`。
- 排查用 `memory_allocated()`、`memory_summary()` 和显存快照；`nvidia-smi` 包含缓存分配器保留的显存，`empty_cache()` 解决不了真正的泄漏。

Q20：***如何对 PyTorch 程序做性能分析（profiling）？***

- **先粗后细：先看 GPU 利用率，低的话多半是数据加载、CPU 开销或频繁同步；再用 `torch.profiler` 看每个算子的耗时和显存，导出时间线分析。**
- 坑：CUDA 是异步的，手动计时前后要 `torch.cuda.synchronize()`，并跳过 warmup；`.item()`、打印张量等隐式同步会打断流水线。
- 优化方向：混合精度、`torch.compile`、更大 batch、FlashAttention、减少拷贝和同步。

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

Q21：***PyTorch 的 hook 有哪些？分别用来做什么？***

- **两类：模块 hook 挂在前向前、前向后或反向时，用来改输入、取中间特征或看梯度（如 Grad-CAM）；张量 hook 在该张量梯度算出后调用，可读取或修改梯度（如梯度反转）。**
- DDP/FSDP 内部也靠 hook 做梯度通信。
- 坑：hook 里保存未 detach 的张量会泄漏显存；用完要 `remove()`。

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
