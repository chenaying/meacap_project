# 编码问题修复指南

## 问题症状

运行 `inference.py` 时出现：
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd3 in position 3510: invalid continuation byte
```

## 快速修复

### 方法 1: 使用自动修复脚本（推荐）

```bash
# 修复所有 Python 文件
python fix_all_encoding.py

# 或者只修复特定文件
python fix_all_encoding.py inference.py utils/ifcap_ef_integration.py args.py
```

### 方法 2: 手动修复特定文件

如果知道问题文件，可以手动修复：

```bash
# 使用 iconv
iconv -f GBK -t UTF-8 problem_file.py > problem_file_utf8.py
mv problem_file_utf8.py problem_file.py

# 或使用 Python
python -c "
with open('problem_file.py', 'rb') as f:
    data = f.read()
try:
    text = data.decode('gbk')
except:
    text = data.decode('gb2312')
with open('problem_file.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed!')
"
```

## 已修复的文件

- ✅ `inference.py` - 已修复缩进错误和中文注释
- ✅ `utils/ifcap_ef_integration.py` - 已移除中文注释

## 诊断步骤

### 1. 检查语法

```bash
python -m py_compile inference.py
```

### 2. 检查编码

```bash
python check_encoding.py
```

### 3. 测试导入

```bash
# 测试基本导入
python -c "import torch; print('Torch OK')"

# 测试模块导入
python -c "from args import get_args; print('Args OK')"
python -c "from utils.ifcap_ef_integration import retrieve_concepts_with_ef; print('EF OK')"
```

## 如果问题仍然存在

### 1. 检查环境变量

```bash
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 2. 检查其他可能的问题文件

常见的问题文件位置：
- `utils/detect_utils.py`
- `utils/generate_utils_.py`
- `utils/some_utils.py`
- `models/clip_utils.py`
- `dataset/ImgDataset.py`

### 3. 使用 Python 的编码错误处理

在 `inference.py` 开头添加：

```python
import sys
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

## 验证修复

修复后，运行：

```bash
# 测试语法
python -m py_compile inference.py && echo "Syntax OK"

# 测试运行（不实际执行，只检查导入）
python -c "
import sys
sys.path.insert(0, '.')
try:
    from args import get_args
    from utils.ifcap_ef_integration import retrieve_concepts_with_ef
    print('All imports OK!')
except Exception as e:
    print(f'Import error: {e}')
    sys.exit(1)
"
```

## 注意事项

1. **备份文件**: 修复前建议备份
   ```bash
   cp inference.py inference.py.bak
   ```

2. **Git 状态**: 如果使用 Git，检查文件状态
   ```bash
   git status
   git diff inference.py
   ```

3. **编辑器设置**: 确保编辑器使用 UTF-8 编码保存文件

