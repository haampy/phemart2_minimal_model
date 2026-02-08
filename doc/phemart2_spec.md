# PheMART2 技术规格书

## 1. 项目目标

预测基因变异(SNP)与疾病的关联关系。核心是一个检索/排序任务：给定一个变异，在1,563个候选疾病中排序出最相关的疾病。

**关键定义**：
- **变异(Variant)**：蛋白质上的单氨基酸替换（missense mutation），以RefSeq HGVS格式标识，例如 `NM_198576.4(AGRN):c.226G>A (p.Gly76Ser)`
- **疾病(Disease)**：由一组HPO表型术语(traits)的集合定义。两个疾病相同当且仅当其HPO集合相同。共1,563个疾病
- **Trait**：单个HPO表型术语，是异质图的节点。共8,526个trait

## 2. 数据文件

### 2.1 主任务标签

**文件**: `data/main_task/output/expanded_labels.csv`

| 列名 | 含义 |
|------|------|
| snps | 变异ID（HGVS格式） |
| gene_name | 所属基因名 |
| disease_index | 疾病ID（0-1562） |
| confidence | 标注置信度 |
| hpo_ids | 该疾病的HPO术语集合（`\|`分隔） |

- 总行数: 53,237（每行是一个 variant-disease 对）
- 唯一变异: 42,452
- 唯一疾病: 1,563
- **18.4%的变异关联多个疾病**（最多19个），平均1.25个

### 2.2 疾病定义

**文件**: `data/main_task/output/new_disease_to_traits.csv`

| 列名 | 含义 |
|------|------|
| disease_index | 疾病ID（0-1562） |
| representative_name | 疾病名称 |
| hpo_count | 该疾病包含的HPO数量 |
| hpo_ids | HPO术语列表（`\|`分隔） |

- 1,563行，每行定义一个疾病
- 每个疾病是1到68个HPO trait的集合

### 2.3 变异嵌入（DeltaP）

**文件**: `data/variant_data/output/variant_x.csv`

- 索引: 变异ID（HGVS格式）
- 维度: **1280**（ESM-2 mutant embedding - wildtype embedding）
- 行数: ~860K（全量），但训练时只加载各任务需要的子集
- **含义**：捕捉突变对蛋白序列表示空间的扰动

### 2.4 蛋白序列嵌入

**文件**: `data/variant_data/output/gene_local_x_mean.csv`

- 索引: 变异ID（与variant_x对齐）
- 维度: **1280**（ESM-2 蛋白序列的mean pooling）
- 行数: ~860K
- **关键特性**：同一基因的所有变异共享相同的值（因为是基因级别的蛋白序列表示）
- **含义**：蛋白质的整体序列上下文

### 2.5 图节点特征

**基因节点特征**: `data/gene_data/output/gene_global_x.csv`
- 索引: 基因名（如 A1CF, BRCA1）
- 维度: **768**
- 行数: 8,999个基因

**Trait节点特征**: `data/trait_data/output/trait_x.csv`
- 索引: HPO ID（如 HP:0000256）
- 维度: **768**
- 行数: 8,526个trait

### 2.6 图边文件

**Gene-Gene (PPI)**: `data/graph_data/output/gene_to_gene.csv`
- 格式: `gene1,gene2`
- 边数: 171,534

**Gene-Trait (HPO注释)**: `data/graph_data/output/gene_to_trait.csv`
- 格式: `Gene Name,HPO`
- 边数: 143,934

**Trait-Trait (语义相似度)**: `data/graph_data/output/trait_to_trait.csv`
- 格式: `HPO 1,HPO 2`
- 边数: 7,732

构建图时需添加反向边（使用 PyG 的 `ToUndirected`），最终4种边类型：
1. `(gene, to, gene)` — 双向PPI
2. `(gene, to, trait)` — 基因→trait
3. `(trait, rev_to, gene)` — trait→基因（自动添加的反向）
4. `(trait, to, trait)` — 双向trait相似度

### 2.7 Domain任务数据

**标签**: `data/domain_data/processed/domain_labels.csv`
- 格式: `snp,genes,domain_map`
- 行数: 171,441
- domain_map: Pfam结构域ID（0-768），共769个类

**结构域嵌入**: `data/domain_data/processed/domain_embeddings.csv`
- 维度: 770行 × **768**列（每个Pfam域的嵌入）

### 2.8 MVP-REG任务数据

**回归目标**: `data/MVP_data/Variant_phenotype_correlation_matrix/ClinVar_EUR_variant_embedding_dim120_svd.npy`
- 格式: NumPy二进制
- 维度: [N_variants, **120**]（SVD降维后的临床相关性嵌入）
- 使用前需L2归一化为单位向量

**变异ID映射**: `data/MVP_data/Variant_phenotype_correlation_matrix/ClinVar_variants_all_with_ref_EUR.csv`
- 列: `variant_raw,variant`（rsID格式）

### 2.9 FUNC_IMPACT任务数据

**文件**: `data/func_impact_data/processed/func_impact_labels.csv`
- 格式: `snp,gene_id,[8个分数列],[8个mask列]`
- 行数: 141,746
- 8个功能影响分数: CADD_phred, phyloP, GERP++, SIFT, Polyphen2_HDIV, MetaSVM, REVEL, AlphaMissense
- 对应8个mask列（1=有效, 0=缺失）

分层体系：
- **Tier 1** (权重1.0): CADD_phred, phyloP, GERP++, MetaSVM（覆盖率~100%）
- **Tier 2** (权重0.5): SIFT, Polyphen2_HDIV（覆盖率>80%）
- **Tier 3** (权重0.5): REVEL, AlphaMissense（覆盖率10-70%）

## 3. 数据分割策略

### 3.1 全局基因级分割

**所有任务共享同一套基因级分割**。以main task的基因为锚点：

1. 从main task标签中提取所有唯一基因
2. 按基因随机分割为 train(70%) / val(15%) / test(15%)
3. 同一基因的所有变异必须在同一split中
4. **将此基因分割传播到所有辅助任务**：Domain、MVP-REG、FUNC_IMPACT中，若基因在main test set中，则该基因的变异在辅助任务中也归为test
5. 辅助任务中存在但main task不存在的基因：独立按基因随机分割

**目的**：防止跨任务数据泄漏 — 不允许test基因的变异在任何任务的训练中出现。

### 3.2 一个变异关联多个疾病的处理

在基因级分割下，同一变异的所有(variant, disease)对都在同一split中（因为变异属于同一基因）。

## 4. 模型架构

### 4.1 整体结构

```
输入层:
  variant_x [B, 1280]        → VariantEncoder    → [B, 256] → [B, 128]
  gene_local_x [B, 1280]     → ProteinSeqEncoder → [B, 256] → [B, 128]
  gene_graph_emb [B, 128]    ← HGT图模型输出（按gene_id索引）

融合层:
  TrilinearFusion([B,128], [B,128], [B,128]) → [B, 128]

任务头:
  Main:        fusion → clip_variant_proj → [B, 128]  ⟷  disease_emb [1563, 128]
  Domain:      fusion → domain_variant_proj → [B, 128] → 与769个domain原型对比
  MVP-REG:     fusion → mvp_regression_head → [B, 120]
  FUNC_IMPACT: fusion → func_impact_head → [B, 8]
```

### 4.2 编码器

**VariantEncoder** 和 **ProteinSequenceEncoder** 结构相同：
```
Linear(1280, 256) → LayerNorm(256) → ReLU → Dropout(p) → Linear(256, 128) → LayerNorm(128)
```
- 输入: 1280d（DeltaP 或 蛋白序列嵌入）
- 隐层: 256
- 输出: 128

### 4.3 HGT图模型

**输入处理**：
```
HeteroDictLinear(gene:768→256, trait:768→256)   # 将不同维度的节点特征投影到统一维度
→ HeteroDictBatchNorm(256)                       # 按节点类型的BatchNorm
→ HGT(hidden=256, out=128, heads=2, layers=2)    # 异质图Transformer
  - 每层: HGTConv + 残差连接 + activation + dropout
  - 输出投影: gene→Linear(256,128), trait→Linear(256,128)
```

**输出**：
- gene_graph_emb: [8999, 128] — 所有基因的图嵌入
- trait_graph_emb: [8526, 128] — 所有trait的图嵌入

**训练时**：每个batch重新计算完整图的前向传播，梯度正常回传到HGT参数。
**验证/测试时**：每个epoch计算一次并缓存（`torch.no_grad()`），所有batch复用。

### 4.4 DiseaseEncoder（疾病嵌入聚合）

疾病 = HPO trait集合。Disease embedding 通过聚合其成员trait的图嵌入生成：

```
输入: trait_graph_emb[trait_ids]  # 取出该疾病所有trait的嵌入 [K, 128]
→ shared_mlp: Linear(128,256)→LayerNorm→ReLU→Dropout→Linear(256,128)→LayerNorm  # [K, 128]
→ Attention Pooling:
    attention_net: Linear(128,32)→LayerNorm→GELU→Dropout→Linear(32,1)  # [K, 1]
    weights = softmax(attention_logits / sqrt(128))                     # [K, 1]
    output = sum(weights * mlp_output)                                 # [128]
```

Disease Encoder已有attention熵正则化（`entropy_weight=0.01`），防止attention过度集中。

**全部疾病嵌入**: 对1,563个疾病分别聚合 → [1563, 128]

### 4.5 TrilinearFusion（三模态门控融合）

```python
# 三个分支分别投影+归一化
variant_norm  = LayerNorm(Linear(128 → 128)(variant_emb))    # [B, 128]
protein_norm  = LayerNorm(Linear(128 → 128)(protein_emb))    # [B, 128]
gene_graph_norm = LayerNorm(Linear(128 → 128)(gene_graph_emb)) # [B, 128]

# 门控网络
concat = cat([variant_norm, protein_norm, gene_graph_norm])  # [B, 384]
gate_logits = MLP(384→32→3)(concat)                          # [B, 3]
gate_weights = softmax(gate_logits / temperature)            # [B, 3]

# 初始bias: [0.1, 0.0, -0.1]（轻微偏向variant）
# 温度: 初始5.0，每epoch乘以0.9衰减，最小0.1

# 加权融合
fusion = gate_weights[:,0:1]*variant + gate_weights[:,1:2]*protein + gate_weights[:,2:3]*gene_graph
→ final_transform: Linear(128,128)→LayerNorm→ReLU→Dropout→Linear(128,128) → [B, 128]
```

### 4.6 投影头（CLIP空间）

**clip_variant_proj** 和 **clip_disease_proj** 结构相同：
```
Linear(128, 256) → LayerNorm(256) → ReLU → Dropout(p) → Linear(256, 128)
```

### 4.7 Domain任务头

```
domain_variant_proj: Linear(128,256)→LayerNorm→ReLU→Dropout→Linear(256,128)
domain_classifier: Linear(128,256)→LayerNorm→ReLU→Dropout→Linear(256,769)
```

但实际使用的是**对比学习方式**：
```python
domain_prototypes = domain_transformation(domain_embeddings)  # [769, 768] → [769, 128]
logits = matmul(domain_variant_proj(fusion), domain_prototypes.T) / temperature
loss = CrossEntropy(logits, labels)
```

其中 `domain_transformation` 是将768维domain嵌入投射到128维的线性层。

### 4.8 MVP-REG回归头

**MVPRegressionHead**: 多头回归结构
```python
# 4个独立的回归头，每个输出30维
heads = [
    Linear(128,256)→LayerNorm→ReLU→Dropout(0.3)→Linear(256,30)
    for _ in range(4)
]
# 输出: cat(head_1, head_2, head_3, head_4) → [B, 120]
```

### 4.9 FUNC_IMPACT预测头

```
Linear(128, 256) → ReLU → Dropout(p) → Linear(256, 8)
```

## 5. 训练流程

### 5.1 多任务训练循环

每个epoch的一个training step：

```python
# 1. 计算共享图嵌入（每batch重新计算，有梯度）
gene_graph_emb, trait_graph_emb = HGT(graph)           # 全图前向
disease_embeddings = DiseaseEncoder(trait_graph_emb)     # 1563个疾病嵌入

# 2. 各任务取各自的batch，计算loss
loss_main    = main_task_loss(main_batch, ...)       × 1.0
loss_domain  = domain_task_loss(domain_batch, ...)   × 0.4
loss_mvp_reg = mvp_reg_loss(mvp_reg_batch, ...)      × 0.05
loss_func    = func_impact_loss(func_batch, ...)     × 0.05

# 3. 加权总loss
total_loss = loss_main + loss_domain + loss_mvp_reg + loss_func

# 4. 反向传播
total_loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 5.2 主任务Loss：Multi-Positive对比学习

**重要**：一个变异可以关联多个疾病（18.4%的变异有多个关联）。Loss必须处理这种多标签情况。

```python
# 编码
clip_emb = normalize(clip_variant_proj(fusion_emb))     # [B, 128]
disease_emb = normalize(clip_disease_proj(disease_raw))   # [B, 128]

# 相似度矩阵
logits = clip_emb @ disease_emb.T / temperature          # [B, B]
# temperature = 0.15

# 构建多正样本mask：
# positive_mask[i,j] = 1 如果 batch中第i个variant 和 第j个variant 关联了相同的disease
# 或者 第i个variant 确实与 第j个disease 关联
# 具体实现：对batch中每个(variant, disease)对，检查是否有其他样本共享variant或disease

# Multi-positive InfoNCE:
# 对每行i，正样本集合 P_i = {j | positive_mask[i,j]=1}
# loss_i = -log( sum_j∈P_i exp(logits[i,j]) / sum_k exp(logits[i,k]) )
# 等价于：将同一variant的所有disease都视为正样本，避免假负样本惩罚
```

### 5.3 Domain任务Loss

```python
fusion_emb = encode_variants(variant_ids, gene_ids, ...)     # [B, 128]
variant_proj = domain_variant_proj(fusion_emb)                 # [B, 128]
domain_protos = domain_transformation(domain_embeddings_768d)  # [769, 128]
logits = variant_proj @ domain_protos.T / temperature          # [B, 769]
loss = CrossEntropy(logits, domain_labels, weight=class_weights)
```

class_weights 是基于类频率的逆权重，用于处理类不平衡。

### 5.4 MVP-REG Loss

```python
fusion_emb = encode_variants(variant_ids, gene_ids, ...)  # [B, 128]
predictions = mvp_regression_head(fusion_emb)               # [B, 120]
predictions_norm = L2_normalize(predictions)                # 归一化到单位球
# targets已预先L2归一化
loss = 1.0 - cosine_similarity(predictions_norm, targets).mean()
```

### 5.5 FUNC_IMPACT Loss

```python
fusion_emb = encode_variants(variant_ids, gene_ids, ...)  # [B, 128]
predictions = func_impact_head(fusion_emb)                  # [B, 8]
targets, valid_mask = batch["targets"], batch["valid_mask"] # [B,8], [B,8]

# column_weights: tier1=1.0, tier2=0.5, tier3=0.5
squared_errors = (predictions - targets)^2 * valid_mask * column_weights
loss = squared_errors.sum() / (valid_mask * column_weights).sum()
```

### 5.6 优化器和调度器

```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
```

### 5.7 早停

- 监控指标: 主任务验证集的 MRR（Mean Reciprocal Rank）
- Patience: 20 epochs
- 每个epoch结束时评估验证集

### 5.8 其他训练参数

```
MAX_EPOCHS = 200
BATCH_SIZE = 128
SEED = 42
GRADIENT_CLIP_NORM = 1.0
DROPOUT = 0.3
HIDDEN_CHANNELS = 256
OUT_CHANNELS = 128
NUM_HEADS = 2 (HGT)
NUM_GRAPH_LAYERS = 2
TEMPERATURE = 0.15 (CLIP)
```

## 6. 评估

### 6.1 主任务评估

在验证/测试集上：

1. 预计算所有1,563个疾病嵌入 → [1563, 128]
2. 预计算所有验证/测试变异嵌入 → [N, 128]
3. 计算相似度矩阵 → [N, 1563]
4. 对每个变异，按相似度排序所有疾病

**指标**：
- **MRR**：真实疾病的排名的倒数的平均值（主要指标）
- **Recall@k** (k=1,5,10,50)：真实疾病出现在top-k中的比例
- **AUROC/AUPRC**：抽样负例（每个正例采样15个负例），计算AUC
  - coarse版本：随机负例
  - hard版本：选择HPO trait重叠度高的疾病作为困难负例
- **分层MRR**：按疾病频率分桶（frequent/medium/rare）分别计算MRR

### 6.2 Domain任务评估

```python
logits = variant_proj @ domain_protos.T  # [N_val, 769]
top1_acc = (argmax(logits) == labels).mean()
top5_acc = (labels in topk(5, logits)).mean()
macro_f1 = f1_score(labels, argmax(logits), average='macro')
```

### 6.3 MVP-REG评估

```python
predictions_norm = L2_normalize(mvp_reg_head(fusion_emb))
cosine = cosine_similarity(predictions_norm, targets).mean()
mae = |predictions_norm - targets|.mean()
mse = (predictions_norm - targets)^2.mean()
loss = 1.0 - cosine
```

### 6.4 FUNC_IMPACT评估

```python
predictions = func_impact_head(fusion_emb)  # [N, 8]
# mask-aware:
mae_overall = (|pred - target| * mask).sum() / mask.sum()
r2 = 1 - SS_res/SS_tot  (仅在mask=1的位置计算)
# 额外: 每个分数的MAE, 每个tier的MAE
```

## 7. 变异编码流程（单个变异的完整前向路径）

```
输入: variant_id, gene_id

1. variant_x = DeltaP_embeddings[variant_id]           # [1280]
2. gene_local_x = protein_seq_embeddings[variant_id]    # [1280]
3. gene_graph_emb = HGT_output_gene[gene_id]            # [128]

4. v = VariantEncoder(variant_x)                         # [128]
5. p = ProteinSeqEncoder(gene_local_x)                   # [128]
6. fusion = TrilinearFusion(v, p, gene_graph_emb)        # [128]

对于主任务:
7. clip_emb = clip_variant_proj(fusion)                   # [128]
8. score(variant, disease_i) = cosine(clip_emb, disease_emb_i) / temperature
```

## 8. 疾病编码流程（单个疾病的完整前向路径）

```
输入: disease_id

1. trait_ids = disease_to_traits[disease_id]  # 例如 [HP:0000256, HP:0001507, ...]
2. trait_embs = trait_graph_emb[trait_ids]     # [K, 128], K为该疾病的trait数

3. transformed = DiseaseEncoder.shared_mlp(trait_embs)  # [K, 128]
4. attn_logits = attention_net(transformed)              # [K, 1]
5. attn_weights = softmax(attn_logits / scale)           # [K, 1]
6. disease_raw = sum(attn_weights * transformed)         # [128]

7. disease_emb = clip_disease_proj(disease_raw)          # [128]
```

## 9. 数据使用方式总结

| 任务 | 训练样本数 | variant_x来源 | gene_local_x来源 | 标签/目标 |
|------|-----------|---------------|------------------|----------|
| Main | ~53K对 | main子集 | main子集 | disease_index |
| Domain | ~171K | domain子集 | domain子集 | domain_map (0-768) |
| MVP-REG | ~136K | mvp子集 | mvp子集 | 120d SVD向量 |
| FUNC_IMPACT | ~141K | mvp子集(共享) | mvp子集(共享) | 8d分数+mask |

- Main/Domain/MVP-REG/FUNC_IMPACT 各自从大的variant_x文件中按需加载对应变异子集
- MVP-REG和FUNC_IMPACT共享同一套MVP变异集合的embeddings
- 所有任务共享同一个HGT图模型和编码器参数
- 每个任务有独立的任务头

## 10. 核心约束

1. **数据分割必须全局一致**：所有任务遵守同一份基因级train/val/test划分
2. **主任务loss必须处理多标签**：18.4%的变异有多个关联疾病，不可将同一变异的其他真实疾病当负例
3. **图的反向边**：构建图时必须添加反向边，确保信息双向流动
4. **训练时图嵌入必须有梯度**：每个batch重新计算HGT前向，不能detach
5. **验证/测试时图嵌入可以缓存**：每epoch计算一次，`no_grad()`
