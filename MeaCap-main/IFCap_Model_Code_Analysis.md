# IFCap 模型代码详细解析文档

## 目录
1. [论文概述](#论文概述)
2. [整体架构](#整体架构)
3. [核心模块详解](#核心模块详解)
4. [检索机制](#检索机制)
5. [训练流程](#训练流程)
6. [推理流程](#推理流程)
7. [关键创新点](#关键创新点)
8. [代码文件结构](#代码文件结构)

---

## 论文概述

**IFCap: Image-like Retrieval and Frequency-based Entity Filtering for Zero-shot Captioning**

IFCap 是一个零样本图像描述生成模型，核心创新包括：
1. **Image-like Retrieval (ILR)**: 从训练集中检索相似的 caption，利用检索信息增强生成
2. **Frequency-based Entity Filtering (EF)**: 基于频率统计过滤实体，生成高质量的 hard prompt
3. **检索增强的前缀映射网络**: 通过 cross-attention 融合检索信息到 soft prompt

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入：图像特征                            │
│                    (batch_size, clip_hidden_size)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              阶段1: Image-like Retrieval (ILR)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 计算图像特征与训练 caption 特征的相似度                │  │
│  │ 2. 检索 top-k 条最相似的 caption                          │  │
│  │ 3. 提取检索 caption 的 CLIP 特征                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  输出: retrieved_features (batch_size, k, clip_hidden_size)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              阶段2: MappingNetwork (前缀映射)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 图像特征投影 → (b, clip_project_length, d_model)      │  │
│  │ 2. Cross-Attention: 图像特征 × 检索特征                   │  │
│  │ 3. 融合可学习前缀                                         │  │
│  │ 4. Transformer 编码                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  输出: soft_prompt (batch_size, prefix_length, gpt_hidden_size) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              阶段3: Entity Filtering (EF) - 可选                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 从检索 caption 中提取实体                              │  │
│  │ 2. 统计实体频率                                           │  │
│  │ 3. 自适应阈值过滤                                         │  │
│  │ 4. 生成 hard prompt: "There are entity1, entity2..."     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  输出: hard_prompt_tokens (batch_size, hard_prompt_length)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              阶段4: 拼接与生成                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ embeddings = [soft_prompt, hard_prompt, caption_tokens] │  │
│  │ 或 [hard_prompt, soft_prompt, caption_tokens]            │  │
│  │ → GPT-2/OPT → 生成 caption                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. MappingNetwork - 检索增强的前缀映射网络

**文件位置**: `IFCap-master/src/ClipCap.py`

#### 1.1 类定义

```python
class MappingNetwork(nn.Module):
    def __init__(
        self,
        clip_project_length: int,    # CLIP 特征投影长度 (默认 10)
        clip_hidden_size: int,       # CLIP 特征维度 (512 或 640)
        prefix_length: int,          # 前缀长度 (默认 10)
        d_model: int,                # GPT 隐藏层维度 (768 for GPT-2)
        num_layers: int = 8,          # Transformer 层数
        num_heads: int = 8,           # 注意力头数
        k: int = 3,                   # 检索 caption 数量
        device: Optional[str] = None
    ):
        super(MappingNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        
        # 图像特征投影层
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        
        # 检索特征投影层
        self.rt_linear = nn.Linear(clip_hidden_size, d_model)
        
        # 可学习前缀嵌入
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, d_model), 
            requires_grad=True
        )
        
        # Transformer 编码器
        self.transformer = Transformer(d_model, num_layers, num_heads)
        
        # Cross-Attention 模块（检索增强）
        self.crossatt = att_gt_n_rt(d_model, 1, num_heads)
```

#### 1.2 前向传播流程

```python
def forward(self, x: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: 图像 CLIP 特征 (batch_size, clip_hidden_size)
        rtf: 检索到的 caption 特征 (batch_size, k, clip_hidden_size)
    
    Returns:
        soft_prompt: (batch_size, prefix_length, d_model)
    """
    # 步骤 1: 图像特征投影
    # (b, clip_hidden_size) → (b, clip_project_length, d_model)
    x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)
    
    # 步骤 2: 检索特征投影
    # (b, k, clip_hidden_size) → (b, k, d_model)
    rtf = self.rt_linear(rtf)
    
    # 步骤 3: Cross-Attention（检索增强）
    # Query: 图像特征, Key/Value: 检索特征
    att_gt_n_rt = self.crossatt(x, rtf)
    # 输出: (b, clip_project_length, d_model)
    
    # 步骤 4: 融合可学习前缀
    prefix = self.prefix_const.unsqueeze(0).expand(
        x.shape[0], *self.prefix_const.shape
    )  # (b, prefix_length, d_model)
    
    inputs = torch.cat((att_gt_n_rt, prefix), dim=1)
    # (b, clip_project_length + prefix_length, d_model)
    
    # 步骤 5: Transformer 编码
    outputs = self.transformer(inputs)
    # 只取前缀部分
    outputs = outputs[:, self.clip_project_length:, :]
    # (b, prefix_length, d_model)
    
    return outputs
```

#### 1.3 Cross-Attention 模块 (`att_gt_n_rt`)

```python
class att_gt_n_rt(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 8, num_heads: int = 8):
        super(att_gt_n_rt, self).__init__()
        self.transformer = Transformer(d_model, num_layers, num_heads)
    
    def forward(self, x: torch.Tensor, rtf: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 图像特征投影 (b, clip_project_length, d_model) - Query
            rtf: 检索特征投影 (b, k, d_model) - Key/Value
        """
        rtf = rtf if rtf is not None else x
        # Transformer 内部使用 cross-attention
        # query=x, key_value=rtf
        outputs = self.transformer(x, rtf)
        return outputs
```

**关键创新**: 使用检索到的 caption 特征作为 Key/Value，图像特征作为 Query，实现检索信息融合。

### 2. MultiHeadAttention - 多头注意力机制

#### 2.1 实现细节

```python
class MultiHeadAttention(nn.Module):
    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, 
                mask: torch.Tensor = None) -> torch.Tensor:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape
        _, m, _ = key_value.shape
        
        # 1. 投影到 Q, K, V
        queries = self.to_queries(query).reshape(
            b, n, self.num_heads, self.head_size
        )
        keys_values = self.to_keys_values(key_value).reshape(
            b, m, 2, self.num_heads, self.head_size
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        
        # 2. 计算注意力分数
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        
        # 3. 应用 mask（如果有）
        if mask is not None:
            attention = attention.masked_fill(
                mask.unsqueeze(dim=3), float("-inf")
            )
        
        # 4. Softmax 归一化
        attention = attention.softmax(dim=2)
        
        # 5. 加权求和
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(
            b, n, d_query
        )
        outputs = self.project(outputs)
        
        return outputs, attention
```

### 3. ClipCaptionModel - 主模型

#### 3.1 类定义

```python
class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        args,
        continuous_length: int = 10,      # soft prompt 长度
        clip_project_length: int = 10,     # CLIP 投影长度
        clip_hidden_size: int = 512,       # CLIP 特征维度
        num_layers: int = 8,               # Transformer 层数
        num_heads: int = 8,                # 注意力头数
        gpt_type: str = 'gpt2',            # 语言模型类型
        soft_prompt_first: bool = False,   # soft prompt 是否在前
        only_hard_prompt: bool = False,    # 是否只用 hard prompt
        k: int = 3                         # 检索数量
    ):
        super(ClipCaptionModel, self).__init__()
        self.soft_prompt_first = soft_prompt_first
        self.only_hard_prompt = only_hard_prompt
        self.continuous_length = continuous_length
        
        # 加载语言模型
        self.gpt, self.gpt_hidden_size = get_language_mode(gpt_type)
        
        # 前缀映射网络
        self.mapping_network = MappingNetwork(
            clip_project_length, clip_hidden_size, continuous_length,
            self.gpt_hidden_size, num_layers, num_heads, k, args.device
        )
```

#### 3.2 前向传播

```python
def forward(
    self,
    continuous_prompt: torch.Tensor,      # 图像 CLIP 特征
    caption_tokens: torch.Tensor,          # caption tokens
    hard_prompts_length: Optional[List] = None,  # hard prompt 长度
    mask: Optional[torch.Tensor] = None,   # attention mask
    retrieved_features: Optional[torch.Tensor] = None  # 检索特征
) -> Tuple[torch.Tensor, ...]:
    # 1. 词嵌入
    caption_embeddings = self.word_embed(caption_tokens)
    
    # 2. 生成 soft prompt（检索增强）
    continuous_embeddings = self.mapping_network(
        continuous_prompt, retrieved_features
    )
    # (b, continuous_length, gpt_hidden_size)
    
    # 3. 拼接 soft prompt 和 hard prompt
    if hard_prompts_length is not None:  # 有 hard prompt
        if self.only_hard_prompt:
            embeddings = caption_embeddings
        elif self.soft_prompt_first:
            # [soft_prompt, hard_prompt + caption]
            embeddings = torch.cat(
                (continuous_embeddings, caption_embeddings), dim=1
            )
        else:
            # [hard_prompt, soft_prompt, caption]
            # 需要按样本分别处理，因为 hard prompt 长度不同
            embeddings = None
            for i in range(len(hard_prompts_length)):
                length = hard_prompts_length[i]
                temp_embeddings = torch.cat((
                    caption_embeddings[i][:length],      # hard prompt
                    continuous_embeddings[i],            # soft prompt
                    caption_embeddings[i][length:]       # caption
                ), dim=0).unsqueeze(dim=0)
                if embeddings is None:
                    embeddings = temp_embeddings
                else:
                    embeddings = torch.cat(
                        (embeddings, temp_embeddings), dim=0
                    )
    else:  # 没有 hard prompt
        embeddings = torch.cat(
            (continuous_embeddings, caption_embeddings), dim=1
        )
    
    # 4. 输入语言模型生成
    out = self.gpt(
        inputs_embeds=embeddings.type(self.gpt.dtype),
        attention_mask=mask
    )
    
    return out
```

---

## 检索机制

### Image-like Retrieval (ILR)

**文件位置**: `IFCap-master/src/image_like_retrieval.py`

#### 训练阶段检索

```python
def image_like_retrieval_train(train_captions, output_path, caption_features, args):
    """
    为训练数据生成检索结果
    """
    retrieved_captions = {}
    
    # 1. 对 caption 特征添加噪声（数据增强）
    noise_features = noise_injection(
        caption_features,
        variance=args.variance,
        device=args.device
    ).to(torch.float16)
    
    # 2. 对每个 caption，检索最相似的 k 条
    for i in tqdm(range(noise_features.shape[0])):
        noise_feature = noise_features[i].unsqueeze(0)
        
        # 计算相似度
        similarity = noise_feature @ caption_features.T
        
        # 排除自身
        similarity[0][i] = 0
        
        # 检索 top-k
        niber = []
        for _ in range(args.K):
            _, max_id = torch.max(similarity, dim=1)
            niber.append(max_id.item())
            similarity[0][max_id.item()] = 0  # 已选中的设为 0
        
        retrieved_captions[train_captions[i]] = [
            train_captions[k] for k in niber
        ]
    
    # 3. 保存结果
    with open(output_path, 'w') as f:
        json.dump(retrieved_captions, f, indent=4)
```

#### 测试阶段检索

```python
def retrieve_caption_test(image_path, annotations, train_captions, 
                          output_path, caption_features, args):
    """
    为测试图像检索相似的 caption
    """
    # 1. 提取图像特征
    image_features = []
    if args.video:
        # 视频：平均帧特征
        for video in tqdm(image_ids):
            video_feature = clip_model.encode_image(...)
            video_feature = torch.mean(video_feature, dim=0)
            image_features.append(video_feature)
    else:
        # 图像：直接编码
        for idx in tqdm(range(0, len(image_ids), bs)):
            image_features.append(
                clip_model.encode_image(...)
            )
    
    image_features = torch.concat(image_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 2. 检索 top-k 相似 caption
    retrieved_captions = {}
    for i in tqdm(range(image_features.shape[0])):
        image_feature = image_features[i].unsqueeze(0)
        similarity = image_feature @ caption_features.T
        
        niber = []
        for _ in range(args.L):  # L 是检索数量
            _, max_id = torch.max(similarity, dim=1)
            niber.append(max_id.item())
            similarity[0][max_id.item()] = 0
        
        retrieved_captions[image_ids[i]] = [
            train_captions[k] for k in niber
        ]
    
    # 3. 保存结果
    with open(output_path, 'w') as f:
        json.dump(retrieved_captions, f, indent=4)
```

---

## 训练流程

**文件位置**: `IFCap-master/src/main.py`

### 训练主循环

```python
def train(args, datasets: CaptionsDataset, model: ClipCaptionModel, ...):
    device = args.device
    model = model.to(device)
    model.train()
    
    # 加载 CLIP 编码器（如果未使用预提取特征）
    if not args.using_clip_features:
        encoder, _ = clip.load(args.clip_model, device=device)
        encoder.eval()
    
    # 优化器设置
    optimizer = AdamW(model.parameters(), lr=args.lr)
    dataloader = DataLoader(...)
    scheduler = get_linear_schedule_with_warmup(...)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    for epoch in range(epochs):
        for idx, (captions_clip, captions_gpt_tokens, 
                  captions_tokens_for_loss, masks, 
                  hard_prompts_length, rt_feat) in enumerate(dataloader):
            
            model.zero_grad()
            
            # 1. 获取 CLIP 特征
            if not args.using_clip_features:
                # 从文本 token 编码
                continuous_prefix = encoder.encode_text(
                    captions_clip_tokens
                ).float()
            else:
                # 使用预提取特征
                continuous_prefix = captions_clip.to(device).float()
            
            # 2. 归一化和噪声注入
            if args.normalize_prefix:
                continuous_prefix /= continuous_prefix.norm(2, dim=-1, keepdim=True)
            continuous_prefix = noise_injection(
                continuous_prefix, 
                variance=args.noise_variance, 
                device=args.device
            )
            
            # 3. 移动到设备
            captions_gpt_tokens = captions_gpt_tokens.to(device)
            captions_tokens_for_loss = captions_tokens_for_loss.to(device)
            masks = masks.to(device)
            rt_feat = rt_feat.to(device)  # 检索特征
            
            # 4. 前向传播
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.using_hard_prompt:
                    outputs = model(
                        continuous_prefix, 
                        captions_gpt_tokens, 
                        hard_prompts_length, 
                        masks, 
                        rt_feat  # 检索特征
                    )
                else:
                    outputs = model(
                        args, continuous_prefix, 
                        captions_gpt_tokens, 
                        mask=masks
                    )
                logits = outputs.logits
            
            # 5. 计算损失
            captions_tokens_for_loss = captions_tokens_for_loss.masked_fill(
                captions_tokens_for_loss == tokenizer.eos_token_id, 0
            )
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                captions_tokens_for_loss.flatten(),
                ignore_index=0
            )
            
            # 6. 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bs` | 80 | 批次大小 |
| `--lr` | 2e-5 | 学习率 |
| `--epochs` | 1 | 训练轮数 |
| `--k` | 5 | 检索 caption 数量 |
| `--clip_project_length` | 10 | CLIP 投影长度 |
| `--continuous_prompt_length` | 10 | Soft prompt 长度 |
| `--num_layers` | 8 | Transformer 层数 |
| `--noise_variance` | 0.016 | 噪声方差 |
| `--using_hard_prompt` | True | 是否使用 hard prompt |
| `--soft_prompt_first` | True | Soft prompt 是否在前 |

---

## 推理流程

**文件位置**: `IFCap-master/src/validation.py`

### 推理主循环

```python
def validation_coco_flickr30k(args, ...):
    # 1. 加载数据
    if args.using_image_features:
        annotations = pickle.load(infile)
    else:
        annotations = json.load(infile)
    
    # 2. 加载检索结果和实体频率（如果使用 EF）
    if args.entity_filtering:
        with open(f'annotations/retrieved_entity/{args.retrieved_info}', 'r') as f:
            retrieved_entities = json.load(f)
    
    with open(f'annotations/retrieved_sentences/{args.retrieved_info}', 'r') as f:
        eval_rt = json.load(f)
    
    # 3. 将检索到的 caption 转换为 CLIP 特征
    test_rt_caps = [i for _, caps in eval_rt.items() for i in caps[:args.k]]
    rt_feats = []
    for idx in range(0, len(test_rt_caps), bs):
        caps = test_rt_caps[idx:idx+bs]
        with torch.no_grad():
            rt_feat_batch = encoder.encode_text(
                clip.tokenize(caps).to(device)
            )
            rt_feats.append(rt_feat_batch)
    
    rt_feats = torch.concat(rt_feats)
    eval_rt_dic = {}
    for image_id, idx in zip(test_rt_id, range(0, len(rt_feats), args.k)):
        eval_rt_dic[image_id] = rt_feats[idx:idx+args.k]
    
    # 4. 对每个图像生成 caption
    for idx, item in tqdm(enumerate(annotations), total=len(annotations)):
        # 4.1 提取图像特征
        if args.using_image_features:
            image_id, image_features, captions = item
            image_features = image_features.float().unsqueeze(dim=0).to(device)
        else:
            image_id = item
            image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
            image_features = encoder.encode_image(image).float()
        
        image_features /= image_features.norm(2, dim=-1, keepdim=True)
        
        # 4.2 获取检索特征
        rt_feats = eval_rt_dic[image_id]
        rt_feats = rt_feats.to(torch.float32)
        
        # 4.3 生成 soft prompt
        continuous_embeddings = model.mapping_network(
            image_features, rt_feats.unsqueeze(0)
        ).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        
        # 4.4 生成 hard prompt（如果使用 EF）
        if args.using_hard_prompt:
            key = image_id
            if not args.entity_filtering:
                # 使用图像特征相似度
                logits = image_text_simiarlity(...)
                detected_objects, _ = top_k_categories(...)
            else:
                # 使用 EF 模块
                if args.adaptive_ef == 'log_normal':
                    detected_objects = log_normal(
                        retrieved_entities[key], args.K
                    )
                elif args.adaptive_ef == 'normal':
                    detected_objects = normal(
                        retrieved_entities[key], args.K
                    )
                else:
                    # 固定阈值
                    detected_objects = list(
                        filter(lambda x: x[0] >= args.K, retrieved_entities[key])
                    )
                    detected_objects = [l[1] for l in detected_objects]
            
            discrete_tokens = compose_discrete_prompts(
                tokenizer, detected_objects
            ).unsqueeze(dim=0).to(args.device)
            discrete_embeddings = model.word_embed(discrete_tokens)
            
            # 拼接
            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat(
                    (continuous_embeddings, discrete_embeddings), dim=1
                )
            else:
                embeddings = torch.cat(
                    (discrete_embeddings, continuous_embeddings), dim=1
                )
        else:
            embeddings = continuous_embeddings
        
        # 4.5 生成 caption
        if 'gpt' in args.language_model:
            if not args.using_greedy_search:
                sentence = beam_search(
                    embeddings=embeddings, 
                    tokenizer=tokenizer, 
                    beam_width=args.beam_width, 
                    model=model.gpt
                )
            else:
                sentence = greedy_search(
                    embeddings=embeddings, 
                    tokenizer=tokenizer, 
                    model=model.gpt
                )
        else:
            sentence = opt_search(...)
```

---

## 关键创新点

### 1. 检索增强的前缀映射

**创新**: 在 MappingNetwork 中使用 cross-attention，让图像特征（Query）关注检索到的 caption 特征（Key/Value），实现检索信息融合。

**代码位置**: `ClipCap.py` 的 `MappingNetwork.forward()`

```python
# 关键代码
att_gt_n_rt = self.crossatt(x, rtf)  # x: 图像特征, rtf: 检索特征
```

### 2. Image-like Retrieval

**创新**: 在训练和测试阶段都使用检索机制，从训练集中检索相似的 caption，利用这些 caption 的特征增强生成。

**代码位置**: `image_like_retrieval.py`

### 3. Frequency-based Entity Filtering

**创新**: 从检索到的 caption 中提取实体，统计频率，使用自适应阈值过滤，生成高质量的 hard prompt。

**代码位置**: `entity_filtering.py`, `utils.py` 的 `log_normal()` 和 `normal()`

### 4. 灵活的 Prompt 组合

**创新**: 支持多种 prompt 组合方式：
- 只有 soft prompt
- 只有 hard prompt
- Soft prompt + Hard prompt（两种顺序）

**代码位置**: `ClipCaptionModel.forward()`

---

## 代码文件结构

```
IFCap-master/src/
├── ClipCap.py                    # 核心模型定义
│   ├── MultiHeadAttention        # 多头注意力
│   ├── TransformerLayer         # Transformer 层
│   ├── Transformer              # Transformer 编码器
│   ├── MappingNetwork           # 前缀映射网络（检索增强）
│   ├── ClipCaptionModel         # 主模型
│   └── ClipCaptionPrefix        # 只训练前缀的版本
│
├── main.py                       # 训练脚本
│   ├── train()                  # 训练主循环
│   └── main()                   # 参数解析和初始化
│
├── validation.py                 # 推理脚本
│   ├── validation_coco_flickr30k()  # COCO/Flickr30k 推理
│   └── validation_nocaps()      # NoCaps 推理
│
├── image_like_retrieval.py       # 检索机制
│   ├── image_like_retrieval_train()  # 训练阶段检索
│   └── retrieve_caption_test()  # 测试阶段检索
│
├── entity_filtering.py           # 实体频率统计
│   └── main()                   # 从检索 caption 提取实体并统计频率
│
├── entities_extraction.py        # 实体提取（训练数据）
│   └── main()                   # 从训练 caption 提取实体
│
├── CaptionsDataset.py            # 数据集类
│   ├── CaptionsDataset          # 数据集定义
│   └── collate()                # 批处理函数
│
├── utils.py                      # 工具函数
│   ├── log_normal()             # 对数正态过滤
│   ├── normal()                 # 正态过滤
│   ├── noise_injection()        # 噪声注入
│   ├── compose_discrete_prompts()  # 生成 hard prompt
│   └── parse_entities()         # 解析实体
│
├── search.py                     # 生成搜索算法
│   ├── greedy_search()          # 贪心搜索
│   ├── beam_search()            # Beam 搜索
│   └── opt_search()             # OPT 模型搜索
│
├── load_annotations.py           # 数据加载工具
│   ├── load_captions()          # 加载 caption
│   ├── load_stopwords()         # 加载停用词
│   └── load_entities_text()    # 加载实体词汇表
│
└── retrieval_categories.py       # 类别检索（用于 hard prompt）
    ├── clip_texts_embeddings()  # 文本特征提取
    ├── image_text_simiarlity()  # 图像-文本相似度
    └── top_k_categories()        # Top-k 类别选择
```

---

## 数据流示例

### 训练阶段

```python
# 1. 输入
captions_clip = (batch_size, clip_hidden_size)  # CLIP 文本特征
rt_feat = (batch_size, k, clip_hidden_size)     # 检索特征

# 2. MappingNetwork
continuous_embeddings = mapping_network(captions_clip, rt_feat)
# 输出: (batch_size, prefix_length, gpt_hidden_size)

# 3. Hard Prompt（如果有）
hard_prompt_tokens = compose_discrete_prompts(filtered_entities)
hard_prompt_embeddings = word_embed(hard_prompt_tokens)
# 输出: (batch_size, hard_prompt_length, gpt_hidden_size)

# 4. 拼接
embeddings = [soft_prompt, hard_prompt, caption_tokens]
# 输出: (batch_size, total_length, gpt_hidden_size)

# 5. GPT-2 生成
logits = gpt(embeddings)
# 输出: (batch_size, total_length, vocab_size)

# 6. 计算损失
loss = cross_entropy(logits, caption_tokens_for_loss)
```

### 推理阶段

```python
# 1. 输入
image_features = (1, clip_hidden_size)  # 图像特征
rt_feat = (1, k, clip_hidden_size)       # 检索特征

# 2. MappingNetwork
continuous_embeddings = mapping_network(image_features, rt_feat)
# 输出: (1, prefix_length, gpt_hidden_size)

# 3. Hard Prompt（如果使用 EF）
detected_objects = log_normal(retrieved_entities[key], alpha=1.0)
discrete_tokens = compose_discrete_prompts(detected_objects)
discrete_embeddings = word_embed(discrete_tokens)

# 4. 拼接
embeddings = [soft_prompt, hard_prompt]

# 5. Beam Search 生成
sentence = beam_search(embeddings, beam_width=5)
# 输出: "A dog is playing in the park."
```

---

## 总结

IFCap 模型通过以下创新实现了零样本图像描述生成：

1. **检索增强**: 利用 Image-like Retrieval 从训练集中检索相似 caption，提供丰富的上下文信息
2. **检索融合**: 在 MappingNetwork 中使用 cross-attention 融合检索信息到 soft prompt
3. **实体过滤**: 通过 Frequency-based Entity Filtering 生成高质量的 hard prompt
4. **灵活组合**: 支持多种 prompt 组合方式，适应不同场景

该模型在零样本设置下取得了优异的性能，特别是在跨域迁移任务上表现突出。

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: IFCap 项目组

