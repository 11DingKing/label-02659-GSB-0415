"""
Pytest配置和共享fixtures
"""
import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def mock_model():
    """模拟NER模型，避免测试时加载真实模型"""
    mock = MagicMock()
    mock.model = MagicMock()
    mock.tokenizer = MagicMock()
    mock.ner_pipeline = MagicMock()
    mock.label_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG"}
    
    # 模拟predict返回
    mock.predict.return_value = [[
        {"entity": "ORG", "word": "UC Berkeley", "score": 0.95, "start": 0, "end": 11}
    ]]
    return mock


@pytest.fixture(scope="module")
def client(mock_model):
    """创建测试客户端，使用模拟模型"""
    with patch('app.main.ner_model', mock_model):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def sample_texts():
    """测试用文本样本"""
    return [
        "CRISPR-Cas9 is a powerful gene editing tool developed at UC Berkeley.",
        "The COVID-19 vaccine was developed by Pfizer and BioNTech.",
        "Albert Einstein worked at Princeton University.",
    ]


@pytest.fixture
def empty_texts():
    """空文本列表"""
    return []


@pytest.fixture
def long_text():
    """超长文本"""
    return "A" * 10000
