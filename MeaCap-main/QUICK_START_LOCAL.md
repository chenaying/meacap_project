# 快速开始 - 使用本地模型路径

## 已知的本地模型路径

根据您提供的信息，CLIP 模型位于：
- **CLIP**: `/home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32`

## 立即运行（使用本地 CLIP 模型）

### 1. 使用本地 CLIP + HuggingFace 其他模型

如果只有 CLIP 是本地路径，其他模型使用 HuggingFace：

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2
```

### 2. 查找其他模型的本地路径

在运行之前，先查找其他模型的本地路径：

```bash
# 方法 1: 运行查找脚本
bash find_local_models.sh

# 方法 2: 手动检查
ls -la /home/cyp/project/mea_cos/MeaCap/checkpoints/
ls -la ~/.cache/huggingface/hub/
```

### 3. 使用全部本地模型（如果都有）

如果所有模型都在本地，使用完整路径：

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32 \
    --parser_checkpoint /home/cyp/project/mea_cos/MeaCap/checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path /home/cyp/project/mea_cos/MeaCap/checkpoints/all-Mini-L6-v2
```

### 4. 使用便捷脚本

我已经创建了 `run_with_local_models.sh` 脚本，您可以根据实际情况修改其中的路径：

```bash
# 编辑脚本，更新模型路径
nano run_with_local_models.sh

# 然后运行
bash run_with_local_models.sh
```

## 使用 EF 模块（本地模型）

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
    --use_memory \
    --use_ef_module \
    --ef_filter_method log_normal \
    --ef_alpha 1.0 \
    --ef_max_concepts 10
```

## 验证路径是否正确

在运行之前，验证路径是否存在：

```bash
# 检查 CLIP 模型
ls -la /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32

# 检查应该能看到类似以下文件：
# - config.json
# - pytorch_model.bin 或 model.safetensors
# - tokenizer.json
# - vocab.json
```

## 常见问题

### 如果路径不存在

1. **检查路径末尾是否有斜杠**：
   ```bash
   # 正确（有斜杠或没有都可以）
   /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32
   /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32/
   ```

2. **检查拼写**：
   ```bash
   # 列出目录内容确认
   ls /home/cyp/project/mea_cos/MeaCap/checkpoints/
   ```

3. **使用绝对路径**：确保使用完整路径，不要使用 `~` 或相对路径（除非明确知道相对位置）

### 如果模型格式不同

如果您的模型文件夹结构不同，可能需要：
- 检查是否有 `config.json` 文件
- 检查是否有 `pytorch_model.bin` 或 `model.safetensors` 文件
- 如果结构不同，可能需要使用 HuggingFace 模型名称而不是本地路径

