# PheMART2 最简重构实现规格（单文件版）

## 1. 实现目标

构建一个代码结构简单、易修改、可训练可测试的多任务系统，保留以下任务：

1. Main：`variant -> disease` 检索（多标签）
2. Domain：`variant -> domain` 分类
3. MVP-REG：`variant -> 120维临床向量` 回归
4. FuncImpact：`variant -> 8维功能分数` 回归（带 mask）

移除任务：

1. MVP-KL（完全不实现）

---

## 2. 设计原则

1. 功能完整，不阉割主流程。
2. 代码简单，不做复杂工程封装（registry/manager/多层context）。
3. 数据处理只保留必要步骤，避免过度包装。
4. 所有任务统一在一个训练循环中执行。
5. 必须避免跨任务数据泄漏。

---

## 3. 数据理解（给实现代码所需的最小语义）

### 3.1 Main 任务

输入标签表（示例列）：

1. `snps`：variant_id（字符串）
2. `gene_name` 或 `gene_id`：基因名
3. `disease_index`：disease_id（整数）
4. `hpo_ids`：`|` 分隔的 HPO 列表

语义：

1. 同一个 variant 可以关联多个 disease（多标签事实）。
2. disease 的表示由对应 HPO trait 集合决定。

### 3.2 Domain 任务

输入标签表（示例列）：

1. `snp`：variant_id
2. `genes` 或 `gene_id`：基因名
3. `domain_map`：类别 id（整数）

### 3.3 MVP-REG 任务

输入：

1. `MVP variant embeddings`：`variant_id -> 1280维`
2. `MVP gene_local embeddings`：`variant_id -> 1280维`
3. `MVP 120维 target`：`variant_id -> 120维目标向量`

语义：

1. 每个 variant 预测一个 120 维连续向量。

### 3.4 FuncImpact 任务

输入标签表（示例列）：

1. `snp`：variant_id（通常是 rsID）
2. `gene_id`
3. 8个分数字段：`CADD_phred, phyloP, GERP++, SIFT, Polyphen2_HDIV, MetaSVM, REVEL, AlphaMissense`
4. 8个 mask 字段：`*_mask`（0/1）

语义：

1. 仅 mask=1 的位置参与回归损失。

### 3.5 图数据

节点特征：

1. `gene_x.csv`：gene 节点特征
2. `trait_x.csv`：trait/HPO 节点特征

边：

1. `gene_to_gene.csv`
2. `gene_to_trait.csv`
3. `trait_to_trait.csv`

disease 到 trait 映射：

1. `disease_to_traits.csv`：`disease_index -> hpo_ids`

---

## 4. 必须执行的数据规则

1. 统一 ID 规范：`str(x).strip().lower()`。
2. 主任务按多标签处理，不能把 batch 内非对角样本一律当负例。
3. 先做全局 variant 划分，再同步到所有任务，避免跨任务泄漏。
4. 图编码在训练时要参与反传，不做 detach 缓存。
5. 图边需要包含反向信息（直接 `ToUndirected` 即可）。

---

## 5. 建议的最简文件结构

1. `config.py`
2. `data.py`
3. `model.py`
4. `losses.py`
5. `eval.py`
6. `train.py`
7. `run.py`

---

## 6. 每个文件该写什么

## 6.1 `config.py`

只定义最小配置（可以是 dataclass 或字典）：

1. 路径
2. 设备
3. 训练超参（lr、batch、epochs、weight decay）
4. 模型维度（hidden、out、num_heads、num_graph_layers）
5. loss 权重（main/domain/mvp_reg/func）
6. split 参数（train/val/test 比例、seed）

不做复杂配置继承，不做实验套件系统。

## 6.2 `data.py`

使用函数式，不建立复杂类。最少函数建议：

1. `normalize_id(x) -> str`
2. `parse_hpo_ids(raw) -> list[str]`
3. `load_main_labels(path) -> DataFrame`
4. `load_domain_labels(path) -> DataFrame`
5. `load_func_labels(path) -> DataFrame`
6. `load_embeddings(path) -> DataFrame`
7. `build_global_variant_split(main_df, domain_df, mvp_df, func_df, seed, ratios) -> dict[variant_id, split]`
8. `apply_split(df, variant_col_or_index, split_map) -> (train_df, val_df, test_df)`
9. `build_mappings(gene_x_df, trait_x_df, disease_df) -> dict`
10. `build_hetero_graph(gene_x, trait_x, edge_files) -> HeteroData`
11. `build_disease_to_traits_map(disease_df, trait_mapping) -> dict[int, list[int]]`
12. `build_variant_positive_map(main_df, variant_mapping) -> dict[int, set[int]]`
13. `make_dataloader_for_task(...)`

数据处理中只做必要步骤：

1. 列名统一（如 `gene_name -> gene_id`, `snp -> variant_id`）。
2. 去空值与无效样本。
3. 通过 shared split map 划分 train/val/test。
4. 将所需字段转成 tensor。

## 6.3 `model.py`

实现一个模型文件，包含：

1. `GraphEncoder`（HGT 或等价异构 GNN）
2. `VariantEncoder`（MLP: 1280 -> d_model）
3. `ProteinEncoder`（MLP: 1280 -> d_model）
4. `Fusion`（三路融合：variant/protein/gene_graph）
5. `DiseaseEncoder`（trait embedding 的 attention pooling）
6. `MultiTaskModel`（统一封装各任务头）

`MultiTaskModel` 需要最少方法：

1. `forward_graph(x_dict, edge_index_dict) -> gene_graph_emb, trait_graph_emb`
2. `encode_variant(variant_ids, gene_ids, variant_x, protein_x, gene_graph_emb) -> z_v`
3. `encode_disease_batch(disease_ids, disease_to_traits, trait_graph_emb) -> z_d`
4. `forward_main(...) -> z_v_main, z_d_main`
5. `forward_domain(...) -> logits_domain`
6. `forward_mvp_reg(...) -> pred_120d`
7. `forward_func(...) -> pred_8d`

---

## 7. 损失函数规范（`losses.py`）

## 7.1 Main（必须是多标签）

输入：

1. `variant_emb`: `[B, d]`
2. `all_disease_emb`: `[D, d]`
3. `positive_disease_ids_per_variant`: `list[set[int]]`

实现：

1. `logits = variant_emb @ all_disease_emb.T / temperature`，shape `[B, D]`
2. 构造 multi-hot target，shape `[B, D]`
3. `BCEWithLogitsLoss`

说明：

1. 不允许 `labels = arange(B)` 的对角 CE。
2. 一个 variant 的多个 disease 都是正样本。

## 7.2 Domain

1. 标准 `CrossEntropyLoss(logits_domain, domain_labels)`。

## 7.3 MVP-REG

1. `pred = normalize(pred, dim=-1)`
2. `target = normalize(target, dim=-1)`
3. `loss = 1 - cosine_similarity(pred, target).mean()`

## 7.4 FuncImpact

1. `err = (pred - target)^2`
2. `loss = (err * mask).sum() / (mask.sum() + eps)`

## 7.5 总损失

1. `loss_total = w_main*L_main + w_domain*L_domain + w_mvp*L_mvp + w_func*L_func`

---

## 8. 训练流程（`train.py`）

保持单一、直接、可读：

1. 初始化 model / optimizer / scaler(可选 AMP)
2. 每个 epoch：
3. 创建各任务迭代器
4. `for step in range(max_steps):`
5. 调用一次 `forward_graph` 得到共享图嵌入
6. 依次尝试获取 main/domain/mvp/func batch
7. 有 batch 就计算对应 loss
8. 聚合总 loss，反向更新
9. epoch 结束后做验证
10. 以 main 的 MRR 作为 best model 保存标准

注意：

1. 训练阶段不要 `detach` 图嵌入。
2. 允许某个任务在某一步没有 batch（直接跳过）。

---

## 9. 评估流程（`eval.py`）

## 9.1 Main 检索指标

1. 先预计算全部 disease embedding：`[D, d]`
2. 对每个样本算 `scores = z_v @ z_d_all.T`
3. 对该样本的所有真实 disease，取最佳排名（best rank）
4. 统计：
5. `MRR`
6. `Recall@1/5/10`

## 9.2 Domain

1. `top1`, `top5`

## 9.3 MVP-REG

1. `cosine`, `MAE`

## 9.4 FuncImpact

1. masked `MAE`

---

## 10. 防泄漏与一致性约束（必须实现）

1. 全局 split 仅生成一次，所有任务共享。
2. 全局 test variants 不允许出现在任意辅助任务训练集中。
3. 主任务评估时，已知真实关联 disease 不能当负样本。
4. 统一 ID 规范化，避免大小写/空格导致错配。
5. 日志中输出每个 split 的样本数与跨任务重叠计数（至少 train/test 重叠检查）。

---

## 11. 最小运行入口（`run.py`）

顺序：

1. 读配置
2. 加载数据
3. 建图与映射
4. 创建 dataloaders
5. 初始化模型
6. 训练
7. 测试并打印最终指标

输出最少内容：

1. 每 epoch：`train_loss`, `val_mrr`, `val_recall@k`
2. 最终：4个任务的测试指标

---

## 12. 验收标准

1. 代码只用上述 7 个文件即可跑通训练与测试。
2. 能同时训练 Main + Domain + MVP-REG + FuncImpact。
3. Main loss 为 multi-positive BCE，而不是对角 CE。
4. 各任务使用共享全局 split，无跨任务 test 泄漏。
5. 输出包含 Main 的 `MRR/Recall@K`。
6. 无 MVP-KL 相关代码路径。

---

## 13. 给 AI 编程工具的执行指令（可直接复制）

请在一个新目录中按以下要求实现代码：

1. 使用 Python + PyTorch + PyG。
2. 按 `config.py/data.py/model.py/losses.py/eval.py/train.py/run.py` 组织代码。
3. 实现 4 个任务：Main、Domain、MVP-REG、FuncImpact。
4. 完全移除 MVP-KL。
5. Main 任务必须用 multi-positive BCE（`[B, D]` logits + multi-hot target）。
6. 先做全局 variant split，再同步到所有任务，避免跨任务泄漏。
7. 训练时图编码参与反向传播，不做 detach 缓存。
8. 评估输出 Main 的 MRR/Recall@1/5/10，Domain 的 top1/top5，MVP-REG 的 cosine/MAE，FuncImpact 的 masked MAE。
9. 代码风格以简单直白为优先，不要引入复杂工程框架。
10. 在关键函数添加简短注释，说明输入输出张量 shape。

