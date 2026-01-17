# Linux 系统运行指南

## 问题说明

`args.py` 中的默认路径是 Windows 路径，在 Linux 系统上无法使用。需要在运行时通过命令行参数指定正确的路径。

## 解决方案

### 方法 1: 使用 HuggingFace 模型名称（推荐）

如果模型已从 HuggingFace 下载，可以直接使用模型名称：

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model openai/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2
```

### 方法 2: 使用 Linux 本地路径

如果模型已下载到 Linux 系统的本地目录（推荐使用本地路径以提高加载速度）：

```bash
# 使用已找到的本地 CLIP 模型路径
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model /home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32 \
    --parser_checkpoint /home/cyp/project/mea_cos/MeaCap/checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path /home/cyp/project/mea_cos/MeaCap/checkpoints/all-Mini-L6-v2
```

**快速查找其他模型路径**：
```bash
# 运行查找脚本
bash find_local_models.sh

# 或者手动检查常见位置
ls -la /home/cyp/project/mea_cos/MeaCap/checkpoints/
ls -la ~/.cache/huggingface/hub/
```

### 方法 3: 修改 args.py 默认值（永久修改）

如果需要永久修改默认路径，编辑 `args.py` 文件：

```python
# 修改第 50-53 行
parser.add_argument('--vl_model', type=str, default='openai/clip-vit-base-patch32')
parser.add_argument("--parser_checkpoint", type=str, default='google/flan-t5-base')
parser.add_argument("--wte_model_path", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument("--lm_model_path", type=str, default='./checkpoints/CBART_COCO')
```

## 完整运行示例

### 使用原始 Parser 方法

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model openai/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2
```

### 使用 EF 模块

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model openai/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
    --use_memory \
    --use_ef_module \
    --ef_filter_method log_normal \
    --ef_alpha 1.0 \
    --ef_max_concepts 10
```

### 使用混合模式（EF + Parser）

```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO \
    --vl_model openai/clip-vit-base-patch32 \
    --parser_checkpoint google/flan-t5-base \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
    --use_memory \
    --use_ef_module \
    --ef_hybrid \
    --ef_filter_method log_normal \
    --ef_alpha 1.0 \
    --ef_max_concepts 10
```

## 常见模型路径

### HuggingFace 模型名称

- **CLIP**: `openai/clip-vit-base-patch32` 或 `openai/clip-vit-large-patch14`
- **Flan-T5**: `google/flan-t5-base` 或 `google/flan-t5-large`
- **SentenceTransformer**: `sentence-transformers/all-MiniLM-L6-v2` 或 `sentence-transformers/all-mpnet-base-v2`

### 检查模型是否存在

```bash
# 检查 HuggingFace 缓存
ls ~/.cache/huggingface/hub/

# 或者使用 Python
python -c "from transformers import CLIPModel; print(CLIPModel.from_pretrained('openai/clip-vit-base-patch32'))"
```

## 环境变量设置（可选）

如果遇到编码问题，可以设置：

```bash
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

## 故障排除

### 1. 模型下载问题

如果模型未下载，HuggingFace 会自动下载，但需要网络连接。如果网络受限，可以：

```bash
# 手动下载模型
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### 2. 路径问题

确保所有路径都是有效的：
- 使用绝对路径或相对于当前目录的路径
- 检查路径是否存在：`ls -la /path/to/model`

### 3. 权限问题

确保有读取模型的权限：
```bash
chmod -R 755 /path/to/models
```

