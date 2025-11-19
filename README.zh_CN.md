# 基于 TileLang 的 DeepSeek-V3.2-Exp DSA Warmup Lightning Indexer 训练算子实现

<p align="center">
  <img src="https://img.shields.io/badge/TileLang-0.1.6.post1+cu126.git7a5077e4-blue" alt="TileLang" style="vertical-align: middle;"/>
  <img src="https://img.shields.io/badge/flash--attn-2.8.3-orange?logo=pypi&logoColor=white" alt="flash-attn" style="vertical-align: middle;"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA_H800_80GB-76B900?logo=nvidia&logoColor=white" alt="GPU" style="vertical-align: middle;"/>
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white" alt="CUDA" style="vertical-align: middle;"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License" style="vertical-align: middle;"/>
</p>

[English](README.md) | 简体中文

- [基于 TileLang 的 DeepSeek-V3.2-Exp DSA Warmup Lightning Indexer 训练算子实现](#基于-tilelang-的-deepseek-v32-exp-dsa-warmup-lightning-indexer-训练算子实现)
  - [1. 最新动态](#1-最新动态)
  - [2. 性能评估](#2-性能评估)
  - [3. 快速开始](#3-快速开始)
  - [4. 设计决策](#4-设计决策)
    - [4.1 背景知识](#41-背景知识)
      - [4.1.1 DSA Warmup Lightning Indexer 算法](#411-dsa-warmup-lightning-indexer-算法)
      - [4.1.2 one-pass KL Divergence fwd/bwd 算法](#412-one-pass-kl-divergence-fwdbwd-算法)
      - [4.1.3 TileLang 即时编译](#413-tilelang-即时编译)
    - [4.2 决策空间](#42-决策空间)
      - [4.2.1 target 分布的设计](#421-target-分布的设计)
      - [4.2.2 算子 Grid 划分 (per head or not)](#422-算子-grid-划分-per-head-or-not)
      - [4.2.3 Lightning Indexer 的数据类型](#423-lightning-indexer-的数据类型)
  - [5. 未来路线](#5-未来路线)
  - [6. 致谢](#6-致谢)


## 1. 最新动态

- 2025/11/19 ✨: 我们很高兴地宣布, ***<u>tl-dsa-warmup-lightning-indexer</u>*** ——基于 [tilelang](https://github.com/tile-ai/tilelang) 的 DeepSeek-V3.2-Exp DSA Warmup Lightning Indexer ***训练算子***, 现已开源!

## 2. 性能评估

- <u>***tl-dsa-warmup-lightning-indexer***</u> 算子与 Flash Attention 兼容, 其 forward pass 同时输出 Flash Attention output 以及 KL Divergence; 其 backward pass 计算 KL Divergence 的梯度
- 以 Flash Attention 为基准, 下表直观地给出了该算子 (经过 tilelang.autotuner.autotune 自动优化) 当前的性能水平

```text
======================================================================================================
  varlen Setting (bs, seq_len)               Fwd Latency                          Bwd Latency
--------------------------------   -------------------------------       -----------------------------
total     batch     seq_len        TL Kernel   flash_attn    Ratio       TL Kernel   flash_attn  Ratio
seq_len   size      qk             (ms)        (ms)                      (ms)        (ms)
======================================================================================================
8K        4         2048           1.99        0.88          2.26x       15.42       3.07        5.02x
16K       8         -              3.96        1.72          2.30x       30.54       5.96        5.12x
32K       16        -              7.81        3.42          2.28x       60.52       11.74       5.16x
64K       32        -              15.59       6.97          2.24x       119.50      23.30       5.13x
128K      64        -              31.05       14.01         2.22x       238.40      46.42       5.14x
======================================================================================================
```

## 3. 快速开始

```bash
python3 kernel_bf16_training_dsa_warmup_lightning_indexer.py --verbose --batch 4
```

## 4. 设计决策

### 4.1 背景知识

#### 4.1.1 DSA Warmup Lightning Indexer 算法

![deepseek_dsa](./images/deepseek_dsa.png)

- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) 的核心算法 DSA 是一个两阶段的算法, 其中第一阶段 (Warmup) 冻结主模型的参数, 仅训练 Lightning Indexer

  - Lightning Indexer 更加轻量: 具有更小的 num_heads, head_dim, 且可以使用 fp8 精度

  - Lightning Indexer 与主模型计算 Logits 的复杂度均为 $$O(N^2)$$ , 但其轻量化的设计使其具有更高效计算 Full Attention 的潜质

  - Lightning Indexer 通过 KL Divergence 实现与主模型计算 Logits 的对齐

  - 经过良好对齐的 Lightning Indexer 可以更高效地计算 ***近似准确*** 的 Logits, 为 DSA 后续的 Top-k Selector 阶段提供输入数据

  - 注: DSA 使用 MLA 的 **MQA** 模式进行训练, 如论文中所述

    > Therefore, we implement DSA based on the MQA mode of MLA.

- 如图所示, 输入 hidden states $$𝐡_t \in ℝ^d$$ 将被投影为 $$𝐪_{t,j}^I \in ℝ^{d^I}$$ , $$𝐤_{t}^I \in ℝ^{d^I}$$ , $$w_{t,j}^I \in ℝ$$

  - 由 Lightning Indexer 计算所得 Logits 表达式为 $$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(𝐪_{tj}^I \cdot 𝐤_s^I\right)$$
  - 即: token $$t$$ 与 token $$s \; (s\leq t)$$ 之间的 Logits $$I_{t,s}$$ 为不同 Indexer head 下 $$\text{Logits}_{t,s}^{h^I}.\text{relu}()$$ ​ 的加权
  - 其中权重系数 $$w_{t,j}^I \in ℝ$$ 由输入 hidden states 投影得到, 用以衡量某一 Indexer head 下特定 query token 的重要度

- Indexer 的训练 Loss 为 $$ℒ^I = \sum_t 𝒟_{\text{KL}}(p_{t,:} \| \text{Softmax}(I_{t,:}))$$

  - 论文中对主模型概率分布 $$p_{t,:}$$ 的相关说明为:

    > To align he indexer outputs with the main attention distribution, for the t-th query token, we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution $$p_{t,:} \in ℝ^t$$

  - 上述过程的表达式可以记为 $$p_{t,:} = \frac{\sum_{h=1}^H A_h[t, :]}{\Vert \sum_{h=1}^H A_h[t, :]\Vert_1}$$ ​, 即 "先 Softmax 后平均"

#### 4.1.2 one-pass KL Divergence fwd/bwd 算法

- KL Divergence 的定义式为 $$𝒟_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$ , 这里的 $$P(i)$$ 或 $$Q(i)$$ 均为 Logits $$p(i)$$ 或 $$q(i)$$ 经过 Softmax 的计算结果。若按定义发来计算 KL Divergence, 则要求 materialize 全部的 Logits
- 按定义法计算 KL Divergence 的方式并不 Memory-efficient, 且与 Flash Attention 的思路相悖。KL Divergence 的实际应用需要与 Flash Attention 兼容, 这便要求 KL Divergence 可以通过对 Query/Key tile 的逐步访问而累积求得, 即具备 one-pass algorithm 的性质
- one-pass KL Divergence 与 flash attention 兼容, 在 flash attention 遍历过程的同时完成 KL Divergence 计算, 同时输出 flash attention 结果与 KL Divergence 结果
- 以下为 ***Tiling*** 版本的输入为 Logits $$\vec{p}\inℝ^N$$ 和 $$\vec{q}\inℝ^N$$ ​ 时 KL Divergence 前向与反向的伪代码。在实际应用中需要进一步推广到 ***二维*** 和 ***varlen*** 形式

![one_pass_dkl_fwd_tiling](./images/one_pass_dkl_fwd_tiling.png)

![one_pass_dkl_bwd_tiling](./images/one_pass_dkl_bwd_tiling.png)

#### 4.1.3 TileLang 即时编译

- TileLang 可以对传入 Tensor 的 batch_size, seq_len_q, seq_len_k 等参数进行自适应, 这在 varlen 场景下极为有用

### 4.2 决策空间

在算子实现过程中, 我们注意到一些可能影响最终实现的设计选择, 总结如下:

#### 4.2.1 target 分布的设计

- 按照论文原文, $$p_{t,:} = \frac{\sum_{h=1}^H A_h[t, :]}{\Vert \sum_{h=1}^H A_h[t, :]\Vert_1}$$ , 令 $$\text{Softmax}(\text{Logits}^\prime_{t, :}) = p_{t,:}$$ 且 $$\text{Softmax}(\text{Logits}^h_{t, :}) = A_h[t, :]$$ , 易知 $$\sum_h \text{Logits}^h_{t, :} \neq \text{Logits}^\prime_{t,:}$$
- 按照上述 one-pass kl divergence fwd/bwd 算法, 需要得知 Softmax 之前的表达式 $$\text{Logits}^\prime_{t,:}$$
- ***<u>在此, 我们选择贯彻 one-pass kl divergence fwd/bwd 算法的思路, 因此不求解</u>*** $$\text{Logits}^\prime_{t,:}$$ ​ ***<u>的具体表达式</u>***
- 一个可选项是使用 $$\sum_h \text{Logits}^h_{t, :}$$

#### 4.2.2 算子 Grid 划分 (per head or not)

- flash attention 相关算子的 Grid 划分, 往往选择按照 batch / max_seq_len / head 的维度进行划分, 即每个 thread group 只处理一个 attention head, per_head 地执行相同逻辑, 以 tilelang 语法为例形式如下:

  ```python
  with T.Kernel(
    T.ceildiv(max_seq_len, block_M),
    heads,
    batch_size,
    threads=num_threads
  ) as (bx, by, bz):
    ...
  ```

- 若使用 $$\sum_h \text{Logits}^h_{t, :}$$ 则需要在同一个 thread group 内处理所有 attention heads, 相应地, flash attention 相关算子的 Grid 划分也将修改为

  ```python
  with T.Kernel(
    T.ceildiv(max_seq_len, block_M),
    batch_size,
    threads=num_threads
  ) as (bx, bz):
    ...
  ```

- 在最初实现该算子的过程中, 我们曾选择了这样的 Grid 布局进行开发, 然而受到这一算法本身合理性与 TileLang 设计的限制, 这种开发方式尚未跑通
  - 算法本身的合理性: 在通常使用的 per head 布局中, 行列维度的 tile 可以开的较大; 而非 per head 布局的行列维度由于 num_heads 这一包袱而不能开的较大, 可能导致潜在的 memory 压力
  - TileLang 设计的限制: 开发该算子所使用的 TileLang 版本为 [0.1.6.post1+cu126.git7a5077e4](https://github.com/tile-ai/tilelang/commit/7a5077e4aa8e30533b6fe1f0716b2c28cf6f661b), 截至开发该算子的时候, TIleLang 设计尚存在局限, 未能良好支持这种开发思路, 相关内容记录在了 [tile-ai / tilelang Issues #1199](https://github.com/tile-ai/tilelang/issues/1199) 中
- ***<u>因此, 我们选择仍使用 per head 布局, 逐 head 地完成 indexer head 与模型 attention head 计算 logits 的对齐</u>***

#### 4.2.3 Lightning Indexer 的数据类型

- 按照论文原文, Lightning Indexer 选择使用 fp8 数据类型, 作为 Lightning Indexer 的效率优势之一
- 实现 Lightning Indexer 的反向传播算子过程中发现 fp8 到 bf16 之间的梯度需额外考虑; 且 Lightning Indexer 可能在训练时使用 bf16 数据类型, 推理时使用 fp8 数据类型
- ***<u>出于开发算子原型的考虑, 我们选择先使用 bf16 开发 Lightning Indexer, 后续支持 fp8 数据类型</u>***



***Finally, 我们的设计选择:***

- ***<u>令 Lightning Indexer 与主模型的 heads 数量相同, 使用 bf16 数据类型逐 head 地对齐 Lightning Indexer 与主模型计算所得的 Logits</u>***
- 尽管 Lightning Indexer 的三大优势 (1) fp8 (2) less num_heads (3) less head_dim 暂时缺失了前两者, 但我们仍保留了 head_dim 较低的优势, 且 kl divergence 与 flash attention 相兼容, 具有 Fast & Memory efficient 的优点

## 5. 未来路线

- [ ] 考虑硬件特性, 改善算子逻辑

- [ ] 完善对 FP8 训练的支持

- [ ] 提供更贴近 DSA 原文的实现, 在同一个线程组内处理所有 attention / indexer heads
- [ ] 更严格的精度验证

## 6. 致谢

- [tile-ai / tilelang](https://github.com/tile-ai/tilelang/tree/main)
- [rockbenben / md-translator](https://github.com/rockbenben/md-translator)
- [svenkreiss / unicodeit](https://github.com/svenkreiss/unicodeit/tree/main)
