# SciBERT NER Service

基于 SciBERT 的专有名词识别（Named Entity Recognition）REST API 服务。

## How to Run

### 一键启动（推荐）

自动检测环境、安装依赖并启动服务：

```bash
# macOS / Linux
./start.sh

# Windows (CMD/PowerShell)
start.bat

# Windows (Git Bash)
bash start.sh
```

### Docker 启动

```bash
# 构建并启动服务
docker-compose up --build -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 本地启动

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Services

| 服务 | 端口 | 描述 |
|------|------|------|
| SciBERT NER API | 8000 | 命名实体识别服务 |

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API 根路径 |
| `/health` | GET | 健康检查 |
| `/ner` | POST | 命名实体识别 |
| `/docs` | GET | Swagger API 文档 |

## 测试账号

本服务为无认证的 REST API，无需账号即可访问。

## 题目内容

使用 SciBERT 实现一个专有名词识别的NER任务，并封装一个REST接口出来。可参考如下代码进行CPU优化：

```python
# BioBERT CPU优化示例
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# 1. 加载量化模型(INT8)
model_name = "dmis-lab/biobert-base-cased-v1.1"
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 多线程配置
torch.set_num_threads(8)  # 设置为CPU核心数

# 3. 推理函数(批量处理优化)
def ner_predict(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        # 解码实体
        predictions = tokenizer.decode(outputs.logits.argmax(dim=2))
        results.extend(predictions)
    return results
```

---

## 项目介绍

### 技术栈

- **后端框架**: FastAPI
- **ML框架**: PyTorch + Transformers
- **模型**: SciBERT (`allenai/scibert_scivocab_cased`)
- **CPU优化**: bitsandbytes (load_in_8bit) + torch动态量化
- **容器化**: Docker + docker-compose

### CPU优化说明

本项目按照Prompt示例代码实现CPU优化，支持两种量化方案：

| 方案 | 条件 | 说明 |
|------|------|------|
| `load_in_8bit` (bitsandbytes) | CUDA + bitsandbytes可用 | ✅ 优先使用（Prompt示例方案） |
| `torch.quantization.quantize_dynamic` | 纯CPU环境 | ✅ 自动回退 |

优化特性：
- INT8量化减少约75%内存占用
- 多线程配置 (`torch.set_num_threads`)
- 批量处理优化
- 推理模式 (`torch.inference_mode`)

### 项目结构

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py      # 配置管理
│   │   ├── main.py        # FastAPI 应用
│   │   ├── model.py       # NER 模型封装
│   │   └── schemas.py     # Pydantic 模型
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py    # Pytest配置和fixtures
│   │   ├── test_api.py    # API集成测试
│   │   ├── test_model.py  # 模型单元测试
│   │   ├── test_schemas.py # Schema验证测试
│   │   └── test_performance.py # 性能测试
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pytest.ini
├── docker-compose.yml
├── start.sh               # 一键启动脚本 (macOS/Linux)
├── start.bat              # 一键启动脚本 (Windows)
└── README.md
```

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| MODEL_NAME | allenai/scibert_scivocab_cased | 模型名称 |
| NUM_THREADS | 4 | CPU 线程数 |
| BATCH_SIZE | 16 | 批处理大小 |
| MAX_LENGTH | 512 | 最大序列长度 |
| PORT | 8000 | 服务端口 |

## API 使用示例

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{"status": "healthy", "model_loaded": true}
```

### 2. 单条文本识别

```bash
curl -X POST "http://localhost:8000/ner" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["CRISPR-Cas9 is a gene editing tool developed at UC Berkeley."]}'
```

响应：
```json
{
  "success": true,
  "data": [
    [
      {"entity": "MISC", "word": "CRISPR-Cas9", "score": 0.9521, "start": 0, "end": 11},
      {"entity": "ORG", "word": "UC Berkeley", "score": 0.8934, "start": 49, "end": 60}
    ]
  ],
  "message": "成功处理 1 条文本，识别出 2 个实体"
}
```

### 3. 批量文本识别

```bash
curl -X POST "http://localhost:8000/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Albert Einstein worked at Princeton University.",
      "The COVID-19 vaccine was developed by Pfizer and BioNTech.",
      "TensorFlow is an open-source machine learning framework by Google."
    ]
  }'
```

### 4. Python 客户端示例

```python
import requests

def ner_predict(texts: list[str], base_url: str = "http://localhost:8000") -> dict:
    """调用NER服务进行实体识别"""
    response = requests.post(f"{base_url}/ner", json={"texts": texts})
    response.raise_for_status()
    return response.json()

# 使用示例
result = ner_predict(["CRISPR-Cas9 is a revolutionary gene editing technology."])
for entities in result["data"]:
    for ent in entities:
        print(f"  - {ent['word']} ({ent['entity']}): {ent['score']:.2%}")
```

### 5. 错误处理

| 状态码 | 错误 | 说明 |
|--------|------|------|
| 400 | Too many texts | 超过100条文本限制 |
| 422 | Validation Error | 请求格式错误 |
| 503 | Model not available | 模型未加载 |
| 500 | Internal server error | 服务器内部错误 |

## 测试

```bash
cd backend

# 运行所有测试
pytest tests/ -v

# 运行测试并生成覆盖率报告
pytest tests/ -v --cov=app --cov-report=html
```

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_api.py` | API端点集成测试 |
| `test_model.py` | NER模型单元测试 |
| `test_schemas.py` | Pydantic模型验证测试 |
| `test_performance.py` | CPU优化和性能测试 |
