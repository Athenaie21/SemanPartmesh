# UNet_predict_angle 结构改进调研（基于 WebSearch）

## 目的

本文整理 `NeurCross/models/UNet.py` 中 `UNet_predict_angle` 的可行结构改进方向。

目标不是直接改代码，而是回答下面几个问题：

- 当前结构的主要瓶颈是什么
- 哪些改进点有公开论文或成熟架构经验支持
- 哪些方向最适合当前这个几何角度预测分支
- 在“参数基本不变 / 适度下降 / 显著下降”三种预算下，优先怎么改

本文结合了本地代码阅读和 WebSearch 检索结果，尽量把“论文里的思路”翻译成“适合当前工程的结构建议”。


## 当前模型的核心特征

当前 `UNet_predict_angle` 的默认结构是：

- `depths=[6, 8, 12, 6]`
- `stem_dims=[512, 512, 1024]`
- `dims=[512, 512, 512, 1024]`
- `decode_out_dims=[512, 512, 512, 256]`

按解析公式估算，当前分支参数量约为：

- 总参数：`29.50M`
- 其中 `decoder` 约占 `71.5%`
- `encoder` 约占 `25.2%`
- 其余模块占比很小

从实现上看，这个分支的主干基本由 `Linear + LayerNorm + ReLU + residual` 组成，核心特征是：

- 沿最后一维 `C` 做通道变换
- token 数量 `N` 在整个过程中保持不变
- 有 skip connection，但没有显式的邻域聚合
- 没有真正的点云层级下采样 / 上采样

因此，当前结构更接近：

- feature-space U-Net
- token-wise residual MLP encoder-decoder

而不是严格意义上的点云层级 U-Net。


## 当前结构的主要问题

### 1. 缺少跨 token 的上下文建模

当前主干几乎全部是逐点 / 逐面 MLP 变换。虽然每个 token 的通道维会变，但点与点、面与面之间并没有在主干中显式交换信息。

这意味着：

- 局部邻域关系没有被直接编码
- 法向、局部坐标系等几何信息虽然输入了，但没有通过邻域结构进一步组织
- 模型很大，但“参数主要花在单点通道映射上”

这也是为什么仅仅继续加宽、加深，很可能不是最优收益方向。


### 2. `decoder` 参数过重

当前大部分参数都堆在 decoder 上，这会带来两个问题：

- 参数预算分配不均衡
- 大量参数用于重复的高维 MLP 解码，而不是用于上下文建模

如果后续要加参数，更合理的思路通常不是继续加 decoder 宽度，而是把一部分预算转移到：

- 局部邻域聚合
- 图消息传递
- 局部注意力
- skip gating


### 3. 结构上是“假 U-Net”，不是“层级 U-Net”

当前网络虽然有 encoder / decoder / skip 结构，但没有：

- 点采样
- 邻域 grouping
- 特征传播
- 感受野逐层扩张

这会限制模型从局部几何到更大尺度几何模式的逐层抽象能力。


### 4. 一些小的结构问题不是主要矛盾

例如：

- 当前有未实际用到的 `Stem`
- `mass` / `faces` 没参与核心计算

这些问题确实值得清理，但它们带来的参数回收非常小，不会决定模型上限。


## WebSearch 调研结论总览

基于公开工作的搜索结果，和当前结构最相关、最值得关注的改进点主要有 8 类：

1. 邻域图消息传递：`DGCNN / EdgeConv`
2. 局部自注意力与相对位置编码：`Point Transformer`
3. 真正的层级点云编码器-解码器：`PointNet++ / PointNeXt`
4. 轻量几何对齐模块：`PointMLP`
5. 深层图网络的稳定扩展：`DeepGCNs`
6. skip connection 重设计：`UNet++ / Attention Gate`
7. 深监督与辅助头：`UNet++ / RFCR / U-Next`
8. MLP 块现代化：`SwiGLU / GEGLU / RMSNorm`

下面逐条展开，并结合当前 `UNet_predict_angle` 给出适配建议。


## 改进点 1：补上局部邻域消息传递

### 代表来源

- [DGCNN / EdgeConv](https://arxiv.org/pdf/1801.07829)
- [Geometric Attentional DGCNN](https://www.sciencedirect.com/science/article/abs/pii/S0925231220319676)

### 检索到的关键结论

`DGCNN` 提出 `EdgeConv` 的出发点非常直接：PointNet 类方法虽然保持了排列不变性，但在局部尺度上仍然“过于独立”，难以捕获点与点之间的几何关系。`EdgeConv` 通过对“点与邻居的边特征”建模，把局部几何关系显式引入网络。

检索原文中明确提到：

- 基础 PointNet 风格方法对每个点独立处理，然后用对称函数汇聚
- 这种独立性会忽略点之间的几何关系
- `EdgeConv` 通过边特征建模局部几何结构，同时保持排列不变性

### 为什么适合当前模型

当前 `UNet_predict_angle` 的最大问题正是：

- 输入是几何量
- 任务也是几何量回归
- 但主干里没有显式的局部几何交互

这与 `EdgeConv` 想解决的问题高度一致。

### 推荐落地方向

适合当前分支的做法不是把整网完全改成 DGCNN，而是局部引入邻域模块：

1. 在输入投影后、进入 encoder 前，加一层或两层 `EdgeConv`
2. 在 bottleneck 前后，各放一个局部图消息传递模块
3. 把 `EdgeConv` 当成 token mixer，用来补 MLP 不擅长的邻域关系建模

### 预期收益

- 让角度预测不再只依赖单 token 特征
- 能利用相邻点 / 面之间的切向一致性、法向变化、局部形变关系
- 对局部方向场、法向相关任务通常更自然

### 风险

- 需要构图，训练和推理开销会增加
- 如果动态图层数过多，工程复杂度会明显上升

### 结论

这是当前结构最值得优先补的能力之一。


## 改进点 2：加入局部自注意力和相对位置编码

### 代表来源

- [Point Transformer (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
- [Point Transformer V2 / V3](https://arxiv.org/html/2312.10035v1)

### 检索到的关键结论

Point Transformer 的检索结果对当前任务有三个特别有用的点：

1. 相对位置编码非常关键  
   原文消融显示：
   - 不加位置编码，效果明显下降
   - 绝对位置编码优于不加
   - 相对位置编码最好
   - 只在一条分支加相对位置编码，不如同时加到 attention 生成和特征变换两条分支

2. 局部自注意力比“单纯邻域 pooling”更强  
   检索片段里比较了：
   - pointwise MLP
   - MLP + neighbor pooling
   - scalar attention
   - vector attention

   结果显示 `vector attention` 明显优于纯 MLP 和普通标量注意力。

3. Point Transformer V2/V3 强调效率问题  
   后续版本引入了更高效的分组向量注意力，以及更高效的邻域组织方式，说明“只要注意力是局部的，就不一定必须走全局 Transformer 那条高成本路线”。

### 为什么适合当前模型

你的输入特征已经包含：

- 点位置
- 法向
- 局部切平面坐标系 `u/v`

这天然适合配合：

- 相对位置编码
- 相对法向差
- 局部框架差异

做局部自注意力。

### 推荐落地方向

1. 不要一上来改成全局 attention
2. 优先考虑局部 `kNN attention`
3. 只在关键位置插入 1 到 2 个 attention block：
   - bottleneck
   - 最终融合头前
4. 如果担心参数和显存，优先考虑分组向量注意力或轻量局部 attention

### 预期收益

- 在不完全改写主干的情况下补上下文建模
- 可以直接建模邻域内方向一致性和边界变化
- 对角度回归这种受局部几何关系约束的任务通常比纯 MLP 更自然

### 风险

- 局部 attention 的收益依赖邻域定义是否合理
- 如果 `N` 很大，仍需控制邻域规模和实现复杂度

### 结论

如果你想在“尽量少改主干”的前提下提高建模能力，这是最强的一条路线之一。


## 改进点 3：把结构真正做成层级点云 U-Net

### 代表来源

- [PointNet++](https://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space)
- [PointNeXt](https://arxiv.org/abs/2206.04670v2)

### 检索到的关键结论

`PointNet++` 的核心价值在于：

- 采样
- grouping
- 局部区域特征提取
- feature propagation

也就是通过层级结构逐步扩大感受野，而不是始终保持所有 token 在同一分辨率下做通道映射。

`PointNeXt` 的检索结果又给了两个重要补充：

- 很多后续方法对 PointNet++ 的超越，并不全来自新模块，训练策略和合理缩放本身也很关键
- 结构上，引入 `inverted residual bottleneck` 和 `separable MLP` 可以更有效地扩展模型

### 为什么适合当前模型

当前分支虽然叫 U-Net，但实际上没有真正的层级抽象。  
如果任务需要：

- 从非常局部的方向一致性
- 到中尺度补丁结构
- 再到更大的全局面片关系

那么真正的层级结构会比“平铺 token + 大量 MLP”更合理。

### 推荐落地方向

1. 长期方案：
   - 把当前 feature-space U-Net 改为 PointNet++ 风格的层级编码-解码器
   - encoder 里做采样和 grouping
   - decoder 里做 feature propagation

2. 中间方案：
   - 保留现有整体框架
   - 只在 encoder 前半部分加入一次下采样 + 局部聚合
   - 在 decoder 恢复分辨率时做一次 feature propagation

3. 结构细节上：
   - 可以参考 PointNeXt 的 inverted residual 和 separable MLP
   - 不必完全复刻原版 PointNet++

### 预期收益

- 感受野随层级自然扩大
- 低层保留局部几何细节，高层提取更抽象结构
- 这种结构在点云 / 几何任务上比纯 MLP U 形结构更有归纳偏置

### 风险

- 改动最大
- 会连带影响数据流和训练脚本接口

### 结论

这是最像“从根上改对结构”的路线，但工程改动也最大，适合做中长期重构。


## 改进点 4：在局部邻域内加入轻量几何对齐或仿射模块

### 代表来源

- [PointMLP](https://arxiv.org/abs/2202.07123v1)

### 检索到的关键结论

PointMLP 的结论对当前项目很有启发：

- 作者认为继续堆复杂局部几何算子，收益可能已经趋于饱和
- 纯 residual MLP 也可以很强
- 但“完全裸奔的深 MLP”在局部几何上会有稳定性问题
- 为此他们引入了一个轻量 `geometric affine module`

检索到的原文片段中明确提到：

- 简单增加深度会降低精度和稳定性
- 不同局部区域的稀疏和不规则几何结构会让共享 residual MLP 难以适配
- 因此需要一个轻量几何仿射模块去处理这种局部差异

### 为什么适合当前模型

你这个分支已经输入了：

- 点位置
- 法向
- 局部坐标系 `u/v`

这说明任务本身已经明确依赖局部几何框架。  
因此，哪怕不引入复杂图卷积或 attention，只要在局部邻域内做一个轻量几何对齐，也很可能比现在的纯 token-wise MLP 更合适。

### 推荐落地方向

1. 在局部邻域构造后，对邻域特征做 affine normalization / transformation
2. 用局部中心点、局部尺度、局部框架去调制邻域特征
3. 不一定需要大模块，轻量几何仿射就可能有效

### 预期收益

- 在不大幅增参的情况下增强几何适配性
- 对局部 frame 相关任务比盲目加深更有针对性

### 风险

- 如果不先引入邻域概念，这类模块的作用会受限

### 结论

如果你想保持“MLP 主干”的整体风格，这是一个非常值得考虑的补丁方向。


## 改进点 5：如果要继续做深层图网络，必须配合稳定化设计

### 代表来源

- [DeepGCNs](https://arxiv.org/abs/1910.06849)

### 检索到的关键结论

DeepGCNs 的核心结论是：

- 图网络如果直接堆深，很容易出现 vanishing gradient 和 over-smoothing
- 残差连接、稠密连接、dilated graph convolution 能显著改善深层训练稳定性
- 文中展示了 7、14、28、56 层图网络：没有 residual 时越深越难训，有 residual 后稳定性明显改善

### 为什么适合当前模型

如果你后续想把当前分支改成：

- graph U-Net
- EdgeConv stack
- 多层图消息传递网络

那么 DeepGCNs 这条经验应该提前纳入设计，而不是“先堆深再看能不能收敛”。

### 推荐落地方向

1. 图模块必须 residual 化
2. 更深时考虑 dense skip 或 stage-level dense concat
3. 用 dilation 或分层邻域扩大感受野，而不是只加层数

### 结论

如果未来结构会往图网络方向发展，这是必须配套吸收的一类经验。


## 改进点 6：重设计 skip connection，而不是裸 concat

### 代表来源

- [UNet++](https://arxiv.org/abs/1912.05074v2)
- [Attention U-Net](https://arxiv.org/pdf/1804.03999)

### 检索到的关键结论

`UNet++` 针对标准 U-Net 提出的两点，和当前结构非常相关：

1. 传统 skip 连接过于“同层直接拼接”，语义差距可能过大
2. 更密集、更渐进的 skip 融合可以减小 encoder 与 decoder 的语义鸿沟

检索结果里提到：

- 标准 U-Net 的 skip 连接限制太强，只允许同尺度特征直接融合
- UNet++ 通过 nested / dense skip 路径来缓解 semantic gap
- deep supervision 还能带来可剪枝性和更好的优化信号

Attention U-Net 的启发则是：

- 不是所有 skip 特征都应该无差别灌给 decoder
- 用 gate 可以压制无关信息、强化有用部分

### 为什么适合当前模型

当前模型的 skip 是最直接的：

- 编码器输出和解码器输出沿最后一维直接拼接
- 没有门控
- 没有渐进融合
- 没有语义对齐

当 encoder 特征较浅、decoder 特征较深时，直接拼接未必是最好融合方式。

### 推荐落地方向

优先级从低风险到高收益可以这样排：

1. 最低风险：加入 skip gate  
   让 decoder 决定保留多少 encoder 信息

2. 中等风险：在 skip 前先做对齐投影  
   先把语义空间对齐，再 concat 或 add

3. 更激进：引入 UNet++ 式 nested skip  
   在 decoder 内部保留更多中间融合节点

### 预期收益

- 降低浅层噪声特征对 decoder 的干扰
- 让多尺度特征融合更平滑
- 可能比继续堆 decoder block 更省参数、更有效

### 结论

对当前结构来说，`skip gating` 是非常值得优先试的低成本改进。


## 改进点 7：加入深监督和辅助头

### 代表来源

- [UNet++](https://arxiv.org/abs/1912.05074v2)
- [RFCR / Omni-supervised Point Cloud Segmentation](https://ar5iv.labs.arxiv.org/html/2105.10203)
- [U-Next](https://arxiv.org/abs/2304.00749)

### 检索到的关键结论

这些工作的共同点是：

- 不只监督最后输出
- 对中间层或多尺度节点加辅助监督
- 用更密的优化信号来改善梯度传播和中间表示质量

RFCR 的检索结果尤其强调：

- 编码器-解码器中间层也可以通过逐层 supervision 获得更明确的表示学习目标
- 深监督有利于 coarse-to-fine 的逐层推理

### 为什么适合当前模型

你的分支最终预测的是逐点 / 逐面的角度值。  
这类任务的一个常见问题是：

- 只有最后一个输出层承担监督
- 中间 decoder 特征学到什么，全靠末端误差信号“反向猜”

如果给中间 decoder 特征增加辅助角度头或局部一致性约束，通常会更稳。

### 推荐落地方向

1. 给 `decode_feat[1]`、`decode_feat[2]`、`decode_feat[-1]` 加轻量辅助头
2. 辅助头可以只在训练时启用
3. 推理时只保留最终输出

### 预期收益

- 梯度路径更短
- 中间特征更有任务针对性
- 对深 decoder 或复杂 skip 结构尤其有帮助

### 风险

- loss 设计需要小心，避免过强辅助监督干扰主目标

### 结论

这是一个低到中等改动、但常常能明显提升训练稳定性的结构增强点。


## 改进点 8：现代化 MLP 块，但把它放在较低优先级

### 代表来源

- [GLU Variants Improve Transformer](https://ar5iv.labs.arxiv.org/html/2002.05202)
- [RMSNorm](https://papers.nips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- [Pre-RMSNorm and Pre-CRMSNorm Transformers](https://openreview.net/pdf?id=z06npyCwDq)

### 检索到的关键结论

现代 FFN / MLP 设计里，有两类常见改进：

1. 激活与门控升级  
   如 `GEGLU` / `SwiGLU`

2. 归一化简化  
   如 `RMSNorm`

检索结果里提到：

- `SwiGLU` / `GEGLU` 通常优于普通 ReLU / GELU
- `RMSNorm` 计算更简单，很多场景下能带来 7% 到 64% 的训练或推理节省

### 为什么它不是第一优先级

因为当前模型的第一瓶颈不是：

- 激活函数不够先进
- 归一化不够省

而是：

- 没有 token mixing
- 没有邻域建模
- 结构不是层级点云编码器

如果先不解决这些核心问题，只把 `ReLU` 换成 `SwiGLU`，收益大概率有限。

### 推荐使用方式

把它当成“第二阶段优化”：

1. 先补上下文建模
2. 再替换 MLP 块内部结构
3. 再考虑 `LayerNorm -> RMSNorm`

### 结论

这是有价值的细化优化，但不应该盖过前面那些更结构性的改进。


## 参数预算角度的建议

结合当前 `29.50M` 参数量，以及上面的结构调研，我更推荐“参数再分配”，而不是单纯继续加宽。

### 方案 A：同预算重分配，优先推荐

思路：

- 把当前深度从 `[6, 8, 12, 6]` 收到 `[4, 6, 8, 4]`
- 保持主宽度 `512 / 1024`
- 把省下来的参数投入到：
  - 2 个局部 attention / graph mixing block
  - 或 1 个 bottleneck attention + 1 个最终融合前 attention

估算结果：

- 收浅后主干约 `21.19M`
- 加 2 个 `1024` 维 attention 后约 `29.59M`

这与当前几乎同预算，但结构会显著更合理。


### 方案 B：中等预算，性价比很高

思路：

- 同样先收浅到 `[4, 6, 8, 4]`
- 只加 1 个 bottleneck 局部 attention
- 再给 skip 加门控

估算结果：

- 总参数约 `26.96M`
- 比当前少约 `8.6%`

这是我认为最稳健的一档。


### 方案 C：轻量化但保留结构升级

思路：

- 主宽度降到 `384 / 768`
- 深度用 `[4, 6, 8, 4]`
- 加 1 个局部 attention
- skip 保留门控

估算结果：

- 总参数约 `15.20M`
- 比当前少约 `48.5%`

如果你怀疑当前模型对数据规模来说过大，或者已经有过拟合迹象，这一档值得认真考虑。


## 改进优先级排序

如果只看“最值得投入时间”的顺序，我建议按下面顺序推进：

1. 先补局部邻域建模  
   `EdgeConv` 或局部 attention，二选一都比继续裸堆 MLP 更优先。

2. 再改 skip 融合  
   至少加入 `skip gate`，避免 encoder 特征直接裸拼接。

3. 再考虑层级化  
   如果愿意做中长期重构，就引入 PointNet++ / PointNeXt 风格层级结构。

4. 再做深监督  
   这是很好的训练稳定器，尤其适合 decoder 较深的版本。

5. 最后才是 MLP 内部现代化  
   如 `SwiGLU` / `RMSNorm`。


## 不建议优先做的事

### 1. 单纯继续加宽

如果没有补 token mixing，再把 `512 / 1024` 扩到更大，参数会明显上升，但不一定触到真正瓶颈。

### 2. 只清理无效模块就期待明显提升

例如删除当前未走到的 `Stem`，参数回收很小，不会本质改变模型能力。

### 3. 先做激活函数微调，忽略结构短板

`SwiGLU`、`RMSNorm` 这类改进是锦上添花，不是雪中送炭。


## 结合当前项目的最终建议

如果目标是“在不大改训练框架的前提下，最有希望提升这个角度分支”，建议优先采用下面这条路径：

1. 把现有深度适度收浅
2. 在 bottleneck 和最终融合前各加一个局部 attention 或图消息传递块
3. 把 skip concat 改成 gated skip fusion
4. 给中间 decoder 节点加轻量深监督

这条路线的优点是：

- 保留当前工程框架
- 不需要一口气重写成全新的 PointNet++ 系统
- 参数可以基本持平，甚至略降
- 但能真正补上当前结构最缺的“上下文建模”

如果目标是“下一版做长期重构”，那就应该直接往：

- PointNet++ / PointNeXt 风格层级结构
- 或 graph U-Net

这两类方向走，而不是把现在这版纯 MLP U-Net 无限加深。


## 参考链接

### 点云局部建模

- DGCNN / EdgeConv: [https://arxiv.org/pdf/1801.07829](https://arxiv.org/pdf/1801.07829)
- Point Transformer: [https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)
- Point Transformer V3: [https://arxiv.org/html/2312.10035v1](https://arxiv.org/html/2312.10035v1)
- PointMLP: [https://arxiv.org/abs/2202.07123v1](https://arxiv.org/abs/2202.07123v1)

### 层级点云结构

- PointNet++: [https://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space](https://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space)
- PointNeXt: [https://arxiv.org/abs/2206.04670v2](https://arxiv.org/abs/2206.04670v2)

### 深层图网络

- DeepGCNs: [https://arxiv.org/abs/1910.06849](https://arxiv.org/abs/1910.06849)

### U-Net 系 skip / supervision

- UNet++: [https://arxiv.org/abs/1912.05074v2](https://arxiv.org/abs/1912.05074v2)
- Attention U-Net: [https://arxiv.org/pdf/1804.03999](https://arxiv.org/pdf/1804.03999)
- RFCR: [https://ar5iv.labs.arxiv.org/html/2105.10203](https://ar5iv.labs.arxiv.org/html/2105.10203)
- U-Next: [https://arxiv.org/abs/2304.00749](https://arxiv.org/abs/2304.00749)

### MLP / Norm 细化改进

- GLU Variants Improve Transformer: [https://ar5iv.labs.arxiv.org/html/2002.05202](https://ar5iv.labs.arxiv.org/html/2002.05202)
- RMSNorm: [https://papers.nips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html](https://papers.nips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- Pre-RMSNorm and Pre-CRMSNorm Transformers: [https://openreview.net/pdf?id=z06npyCwDq](https://openreview.net/pdf?id=z06npyCwDq)


## 一句话结论

当前 `UNet_predict_angle` 最大的问题不是参数不够，而是参数主要花在“逐 token 通道 MLP”上，缺少显式的邻域与层级建模。  
所以最值得优先投入的改进，不是继续堆宽堆深，而是把一部分预算转移到：

- 局部邻域消息传递
- 局部注意力
- skip gating
- 层级点云编码器-解码器

这几类改动更有希望真正提升结构上限。
