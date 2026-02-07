# Polar 位置编码说明

本文档说明 `diffusion_polar.py` 中新位置编码的原理、实现方式和实验方法。

## 1. 设计目标

在原版 `diffusion.py` 里，注意力对 `Q/K` 使用 RoPE（旋转位置编码）。  
新版本 `diffusion_polar.py` 改为 **Polar per-dim gate**，目标是：

1. 保留位置相关性，但降低实现复杂度
2. 让位置调制直接按维度作用到 `Q/K`
3. 引入 MASK 状态门控，让被遮蔽位置在注意力中可控降权

---

## 2. 核心思路

### 2.1 位置门控（pos_gate）

对每个位置 `i`、每个 head 内维度 `k`，定义：

\[
\omega_k = \frac{1}{base^{k / D}}, \quad
gate_{i,k} = \cos(i \cdot \omega_k)
\]

其中 `D = head_dim`。  
预计算后缓存为：

- `pos_gate`: `(1, L, 1, head_dim)`
- 运行时截断到当前序列长度 `T` 后广播到 `(B, T, H, D)`

代码位置：
- `diffusion_polar.py` 中 `_precompute_polar_gate(...)`
- `diffusion_polar.py` 中 `self.register_buffer("pos_gate", ...)`

### 2.2 状态门控（state gate）

根据 token 是否为 MASK（`idx == mask_token_id`）构造对角门控：

\[
a_i =
\begin{cases}
\alpha, & \text{if token is MASK} \\
1.0, & \text{otherwise}
\end{cases}
\]

其中 `alpha = mask_gate_alpha`（默认 `0.3`），形状广播为 `(B, T, 1, 1)`。

### 2.3 可学习相位偏置（phi）

每个 `head_dim` 一组可学习参数 `phi_k`，当前实现采用低成本近似：

\[
\tilde{gate}_{i,k} = \cos(i \cdot \omega_k)\cdot\cos(\phi_k)
\]

最终对 `Q/K` 施加门控：

\[
Q' = Q \odot \tilde{gate} \odot a,\quad
K' = K \odot \tilde{gate} \odot a
\]

再执行 `RMSNorm(Q/K)` 和双向注意力。

---

## 3. 与原 RoPE 的主要区别

1. **作用形式不同**
- RoPE 是二维子空间旋转（成对维度）
- Polar gate 是按维度缩放调制（不要求维度配对）

2. **可学习参数不同**
- 原 RoPE 通常固定频率，无额外相位参数
- Polar 版本额外引入 `phi`，用于学习维度级相位偏置（当前实现为近似形式）

3. **额外状态控制**
- Polar 版本叠加了 MASK 状态门控 `a_i`

---

## 4. 代码结构映射

### 4.1 `Model`

1. 在 `token_emb` 后、`blocks` 前，预计算并缓存 `pos_gate`
2. `forward` 中：
- `pos_gate = self.pos_gate[:, :T]`
- `token_is_mask = (idx == mask_token_id)`
- 传入每个 `Block`

### 4.2 `Block`

`forward(self, x, pos_gate, token_is_mask)`  
把两个门控继续传给 `MultiHeadAttention`。

### 4.3 `MultiHeadAttention`

1. 新增参数：
- `phi: nn.Parameter(torch.zeros(head_dim))`
- `mask_gate_alpha`
2. 新 `forward` 签名：
- `forward(self, x, pos_gate, token_is_mask)`
3. 在 `Q/K` 上应用 `pos_gate * cos(phi)` 和 `state gate a`

---

## 5. 训练与对比方法

### 5.1 单模型训练

```bash
python diffusion_polar.py --train
```

默认权重：
- `weights/diffusion_polar.pt`

### 5.2 RoPE vs Polar 对比

使用脚本：
- `compare_positional_encodings.py`

示例：

```bash
python compare_positional_encodings.py \
  --seeds 1337,2027,9001 \
  --max-iters 3000 \
  --eval-interval 300 \
  --eval-iters 100
```

输出：
1. `ablation_results.json`
2. `ablation_plots/val_loss_curve.png`
3. `ablation_plots/summary_bars.png`
4. 权重（默认保存）：`weights/ablation/{rope|polar}_seed*.pt`

### 5.3 生成质量评估

使用脚本：
- `compare_generation_quality.py`

示例：

```bash
python compare_generation_quality.py \
  --rope-weights weights/ablation/rope_seed1337.pt \
  --polar-weights weights/ablation/polar_seed1337.pt \
  --num-prompts 32 \
  --prompt-len 32 \
  --gen-len 256 \
  --temp 0.8 \
  --top-k 2 \
  --confidence-threshold 0.95 \
  --seed 1337 \
  --output generation_eval_seed1337.json
```

---

## 6. 当前实现的已知边界

1. `phi` 当前是近似门控 `cos(phase)*cos(phi)`，不是严格的 `cos(phase+phi)`。
2. 自动指标（如 `distinct_2`、`repeat_3gram`）不能完全代表可读性，建议配合样本文本人工检查。
3. `mask_gate_alpha`、`confidence_threshold` 对生成稳定性影响较大，建议做小网格调参。

---

## 7. 小结

`diffusion_polar.py` 的新位置编码把“位置调制 + MASK 状态调制”统一作用在 `Q/K` 上，形成一种轻量、可学习、可控的替代 RoPE 方案。  
它适合通过统一 A/B 流程与生成评估脚本做系统对比，再决定在你的目标（质量 / 稳定性 / 速度）下是否优于原 RoPE。
