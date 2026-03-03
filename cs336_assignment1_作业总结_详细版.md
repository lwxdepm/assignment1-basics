# CS336 Assignment 1 (basics) 作业总结（详细版）

> 来源：`cs336_spring2025_assignment1_basics.pdf`（Spring 2025, v1.0.6）
> 目标：完整覆盖作业要求、实现项、实验项、提交与资源约束，不遗漏关键细节。

---

## 0. 全局规则（先看这个）

### 0.1 允许/禁止使用的框架内容
- 作业强调 **from scratch**。
- 你**不能**直接使用 `torch.nn` / `torch.nn.functional` / `torch.optim` 中的现成模块与实现（除指定例外）。
- 允许例外：
  - `torch.nn.Parameter`
  - `torch.nn` 容器类（`Module`、`ModuleList`、`Sequential` 等）
  - `torch.optim.Optimizer` 基类
- 其他 PyTorch 定义原则上可用；不确定时按“是否破坏从零实现精神”判断。

### 0.2 AI 工具政策
- 允许问 LLM：低层编程问题 / 高层概念问题。
- **禁止**直接用 LLM 代做题目。
- 官方建议关掉 AI 自动补全（比如 Copilot Tab）避免浅层完成。

### 0.3 代码结构与测试规范
- 主要代码写在 `cs336_basics/*`。
- `adapters.py` 只做 glue code（调用你的实现），不应包含实质逻辑。
- `test_*.py` 不可修改。
- 典型流程：先补 adapter，再运行对应 `pytest` 子测试。

### 0.4 提交物
- Gradescope：
  - `writeup.pdf`（书面题，建议排版）
  - `code.zip`（你的代码）
- Leaderboard：向 `assignment1-basics-leaderboard` 提 PR（看该仓库 README）。

### 0.5 数据集
- TinyStories（Eldan & Li, 2023）
- OpenWebText（Gokaslan et al., 2019）
- 都是大文本文件（课程机 `/data` 可取，或按 README 下载）。

### 0.6 低资源提示（贯穿全文）
- 官方明确支持 CPU / Apple Silicon (MPS) 下做缩放版实验。
- 建议先用小数据/小模型调通，再上全量。

---

## 1. 作业总览：你最终要做什么

### 1.1 需要你“实现”的组件
1. BPE tokenizer（训练 + 编解码）
2. Transformer LM（含各基础模块）
3. Cross-entropy loss + AdamW
4. Training loop（含 checkpoint 保存/恢复）

### 1.2 需要你“实际跑起来”的流程
1. 在 TinyStories 上训练 BPE tokenizer
2. 用 tokenizer 把数据转成 token IDs
3. 在 TinyStories 上训练 Transformer LM
4. 生成文本并评估 perplexity
5. 在 OWT 上训练并做 leaderboard 提交

---

## 2. Part I：BPE Tokenizer（第 2 章）

## 2.1 Unicode 基础题

### `unicode1`（1 分）
- 问 `chr(0)` 是什么字符；
- `repr` 与 print 表现差异；
- 该字符插入文本时会发生什么。
- 交付：每小问一句话。

### `unicode2`（3 分）
- 为什么 tokenizer 训练更偏向 UTF-8 而不是 UTF-16/32；
- 给错误 UTF-8 解码函数的反例并解释；
- 给一个“无法解码为 Unicode”的 2-byte 序列并解释。

## 2.2 BPE 训练原理与必须遵守的细节

### 核心流程
1. 初始化词表：256 字节 + special tokens
2. 预分词（pre-tokenization）：使用 GPT-2 风格 regex
3. 统计 pair 频次，迭代做 merge 直到达到词表目标

### 关键细节（高频丢分点）
- 不跨 pre-token 边界做 merge。
- 频次 tie-break：选**字典序更大**的 pair。
- special token 要加入词表且有固定 token ID。
- 在 pre-tokenization 之前，要先按 special tokens 把文本切开，防止跨边界合并。
- 推荐 `re.finditer`（避免一次性存预分词结果）。
- 可用 `multiprocessing` 并行预分词。
- merge 阶段可做 pair-count 增量更新优化。

## 2.3 编程题与实验题

### `train_bpe`（15 分）
- 输入：
  - `input_path: str`
  - `vocab_size: int`
  - `special_tokens: list[str]`
- 输出：
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`（按创建顺序）
- 测试：实现 `adapters.run_train_bpe`，跑 `uv run pytest tests/test_train_bpe.py`。

### `train_bpe_tinystories`（2 分）
- TinyStories 上训 10,000 词表，含 `<|endoftext|>`。
- 需序列化 vocab 和 merges。
- 汇报：训练耗时、内存占用、最长 token 及合理性。
- 资源要求：≤30 分钟、≤30GB RAM。
- 提示：合理实现可 <2 分钟。

### `train_bpe_expts_owt`（2 分）
- OWT 上训 32,000 词表，序列化结果。
- 汇报最长 token 及合理性。
- 对比 TinyStories tokenizer 与 OWT tokenizer。
- 资源要求：≤12 小时、≤100GB RAM。

### `tokenizer`（15 分）
实现 `Tokenizer`：
- `__init__(vocab, merges, special_tokens=None)`
- `from_files(...)`
- `encode(text) -> list[int]`
- `encode_iterable(iterable) -> Iterator[int]`（流式、常量内存）
- `decode(ids) -> str`

要求要点：
- encode 按训练得到的 merge 顺序应用。
- special tokens 编码时要按“不可拆分”处理。
- decode 遇到非法 Unicode 用 replacement char（`errors='replace'`）。

### `tokenizer_experiments`（4 分）
1. 采样 TinyStories/OWT 各 10 篇，用两套 tokenizer（10K/32K）比较压缩率（bytes/token）
2. 用 TinyStories tokenizer 编 OWT，比较压缩率并做定性描述
3. 测 tokenizer 吞吐（bytes/s），估算 tokenize 825GB Pile 需时
4. 把 train/dev 编码存成 token IDs（建议 `uint16`），并解释为何合适

---

## 3. Part II：Transformer LM（第 3 章）

## 3.1 架构总览
- 输入 token IDs -> token embedding
- 经过 `num_layers` 个 pre-norm Transformer block
- final norm + output projection 得到 logits
- 训练时按 next-token prediction 做 CE；推理时循环采样。

## 3.2 模块实现清单（带分值）

### `linear`（1 分）
- 自写无 bias `Linear`（继承 `nn.Module`）
- 参数用 `nn.Parameter`
- 注意存的是 `W`（不是 `W^T`）
- 测试：`adapters.run_linear` + `pytest -k test_linear`

### `embedding`（1 分）
- 自写 `Embedding`（不能用 `nn.Embedding`）
- embedding matrix 形状 `(vocab_size, d_model)`
- 测试：`adapters.run_embedding`

### `rmsnorm`（1 分）
- 按 RMSNorm 定义实现
- forward 需先 upcast 到 `float32` 再归一化，最后 cast 回原 dtype
- 测试：`adapters.run_rmsnorm`

### `positionwise_feedforward` / SwiGLU（2 分）
- 形式：`W2(SiLU(W1x) ⊙ W3x)`
- 可直接用 `torch.sigmoid`
- `d_ff` 约 `8/3 * d_model`，且取 64 的倍数
- 测试：`adapters.run_swiglu`

### `rope`（2 分）
- 实现 RoPE，支持任意前置 batch 维
- 输入 `token_positions` 决定位置切片
- 可预计算 sin/cos 用 `register_buffer(persistent=False)`
- 测试：`adapters.run_rope`

### `softmax`（1 分）
- 自写 softmax（减最大值做稳定化）
- 测试：`adapters.run_softmax`

### `scaled_dot_product_attention`（5 分）
- 支持高阶 batch-like 维
- 支持布尔 mask（True 可见，False 不可见）
- 输出中 False 位置概率需为 0，True 位置概率归一
- 测试：`adapters.run_scaled_dot_product_attention`

### `multihead_self_attention`（5 分）
- 实现**因果**多头自注意力（不能看未来）
- RoPE 只用于 Q/K，不用于 V
- `d_k=d_v=d_model/num_heads`
- 测试：`adapters.run_multihead_self_attention`

### `transformer_block`（3 分）
- pre-norm block：
  - `x + MHA(RMSNorm(x))`
  - 再 `+ FFN(RMSNorm(...))`
- 测试：`adapters.run_transformer_block`

### `transformer_lm`（3 分）
- 组装完整 LM：embedding -> N blocks -> final norm -> LM head
- 测试：`adapters.run_transformer_lm`

### `transformer_accounting`（5 分，书面）
- GPT-2 XL 参数量与仅加载参数所需显存
- 前向矩阵乘 FLOPs 分解
- 哪些模块 FLOPs 最大
- GPT-2 small/medium/large 对比 FLOPs 比例
- context 从 1024 到 16384 后 FLOPs 与比例变化

---

## 4. Part III：训练组件（第 4 章）

### `cross_entropy`
- 实现稳定 CE：
  - 减最大值
  - 尽量约掉 log/exp
  - 支持任意前置 batch 维
  - 返回 batch 平均
- 测试：`adapters.run_cross_entropy`

### `learning_rate_tuning`（1 分）
- 跑给定 SGD toy 例子，用 `1e1, 1e2, 1e3`（10 步）
- 观察 loss 是更快下降、变慢还是发散

### `adamw`（2 分）
- 继承 `torch.optim.Optimizer` 实现 AdamW
- 包含一阶/二阶矩、偏置校正、解耦 weight decay
- 测试：`adapters.get_adamw_cls`

### `adamwAccounting`（2 分）
- 训练峰值内存分解：参数/激活/梯度/优化器状态
- 代入 GPT-2 XL（80GB）求可行最大 batch
- 一步 AdamW FLOPs 表达式
- 假设 A100 FP32 峰值 + 50% MFU，估算指定训练配置耗时天数

### `learning_rate_schedule`
- 实现 warmup + cosine anneal + post-anneal 函数
- 测试：`adapters.get_lr_cosine_schedule`

### `gradient_clipping`（1 分）
- 实现全局 L2 norm clipping，`eps=1e-6`
- 原地改参数梯度
- 测试：`adapters.run_gradient_clipping`

---

## 5. Part IV：训练循环与工程化（第 5 章）

### `data_loading`（2 分）
- 输入：token numpy array + `batch_size` + `context_length` + device
- 输出：`(x, y)` 两个 tensor，形状均 `(batch_size, context_length)`
- 都放到指定设备
- 测试：`adapters.run_get_batch`

> 大数据建议：用 `np.memmap` 或 `np.load(..., mmap_mode='r')`，避免全量入内存。

### `checkpointing`（1 分）
- `save_checkpoint(model, optimizer, iteration, out)`
- `load_checkpoint(src, model, optimizer) -> iteration`
- 需完整恢复模型、优化器、步数
- 测试：`adapters.run_save_checkpoint` / `run_load_checkpoint`

### `training_together`（4 分）
- 写主训练脚本，至少支持：
  - 模型/优化器超参可配置
  - 大数据 memmap 加载
  - 定期 checkpoint
  - 定期日志（训练/验证，可接 W&B）

---

## 6. Part V：文本生成（第 6 章）

### `decoding`（3 分）
实现解码函数，支持：
- 给 prompt 续写，直到 `<|endoftext|>` 或达 `max_new_tokens`
- temperature softmax
- top-p（nucleus）采样

---

## 7. Part VI：实验与消融（第 7 章）

## 7.1 实验记录要求

### `experiment_log`（3 分）
- 建立实验追踪基础设施
- 记录 loss 曲线：
  - 横轴至少有 step
  - 且要有 wallclock time
- 交“实验日志文档”（你尝试了什么、效果如何）

## 7.2 TinyStories 训练与分析

### 官方建议基线配置
- `vocab_size=10000`
- `context_length=256`
- `d_model=512`
- `d_ff=1344`（约 8/3 d_model 且 64 对齐）
- `num_layers=4`
- `num_heads=16`
- `rope_theta=10000`
- 总 token 预算约 `327,680,000`

### `learning_rate`（3 分，约 4 H100 小时）
- 做 LR sweep，提交曲线 + 搜索策略
- 目标：TinyStories val loss（per-token）≤ 1.45
- 还要做“稳定边界”分析：包含至少一个发散 run

> 低资源替代：CPU/MPS 可降到 ~40M tokens，目标放宽到 val loss 2.00。

### `batch_size_experiment`（1 分，约 2 H100 小时）
- batch size 从 1 到显存上限（至少含 64 和 128）
- 必要时重新调 LR
- 提交曲线 + 结论

### `generate`（1 分）
- 用训练好的 checkpoint 生成文本
- 交至少 256 tokens（或到第一个 `<|endoftext|>`）
- 评论流畅度 + 至少两个影响质量因素

## 7.3 消融

### `layer_norm_ablation`（1 分，约 1 H100 小时）
- 去掉 RMSNorm 后训练
- 看原最优 LR 是否稳定；能否靠降低 LR 稳定
- 交曲线 + 分析

### `pre_norm_ablation`（1 分，约 1 H100 小时）
- 把 pre-norm 改成 post-norm
- 交与 pre-norm 对比曲线

### `no_pos_emb`（1 分，约 1 H100 小时）
- 去掉位置编码（NoPE）
- 交 RoPE vs NoPE 曲线

### `swiglu_ablation`（1 分，约 1 H100 小时）
- SwiGLU vs SiLU（参数量近似匹配）
- 交曲线 + 简短结论

## 7.4 OWT 主实验与 leaderboard

### `main_experiment`（2 分，约 3 H100 小时）
- 在 OWT 用与 TinyStories 相同架构和训练步数训练
- 交：
  - OWT 学习曲线
  - 与 TinyStories loss 差异解读
  - 生成文本 + 流畅度分析

### `leaderboard`（6 分，约 10 H100 小时预算）
- 规则：
  - 单次运行最多 1.5 小时 H100（如 Slurm 设 `--time=01:30:00`）
  - 只能用提供的 OWT 训练集
- 交付：
  - 最终 val loss
  - wallclock x 轴且总时长 <1.5h 的曲线
  - 方法说明
- 预期：至少优于 naive baseline（loss 5.0）

---

## 8. 评分重心与工作量直觉

- **分值大头**：
  - `train_bpe`、`tokenizer`、`scaled_dot_product_attention`、`multihead_self_attention`
  - `training_together`、`leaderboard`
- **工程大头**：
  - tokenizer 训练性能（预分词并行 + merge 优化）
  - 训练脚本可配置化与可恢复（checkpoint）
  - 实验追踪（曲线、日志、对比）
- **最容易漏的细节**：
  - BPE tie-break 规则（字典序更大）
  - special token 切分后再预分词
  - decode 的 Unicode 错误替换
  - RMSNorm 的 float32 上采样
  - causal mask 方向正确
  - 记录 wallclock-time 轴（实验题明确要求）

---

## 9. 一句话执行路线（建议）

1. 先把 tokenizer 全链路做通并过测试（训练 + encode/decode + 序列化）。
2. 再按模块逐个过 Transformer 子测试（Linear/Embedding/RMSNorm/Attention/Block/LM）。
3. 完成 loss/AdamW/schedule/clipping + dataloader/checkpoint。
4. 跑可恢复训练脚本，先小规模 overfit/debug，再上 TinyStories 全配置。
5. 最后做系统实验、消融、OWT 主实验和 leaderboard。

---

如果你希望，我可以再给你补一版“按时间预算的执行清单”（例如 2 天/3 天冲刺版），把每一步预计耗时和验收标准列出来。