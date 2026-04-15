"""
API端点集成测试
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestHealthEndpoint:
    """健康检查端点测试"""
    
    def test_health_check_success(self, client):
        """测试健康检查返回正常"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_health_response_schema(self, client):
        """测试健康检查响应格式"""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """根路径端点测试"""
    
    def test_root_returns_service_info(self, client):
        """测试根路径返回服务信息"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == "SciBERT NER Service"


class TestNEREndpoint:
    """NER识别端点测试"""
    
    def test_ner_single_text(self, client, mock_model):
        """测试单条文本识别"""
        mock_model.predict.return_value = [[
            {"entity": "ORG", "word": "UC Berkeley", "score": 0.95, "start": 49, "end": 60}
        ]]
        
        response = client.post("/ner", json={
            "texts": ["CRISPR-Cas9 is a gene editing tool developed at UC Berkeley."]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
    
    def test_ner_batch_texts(self, client, mock_model, sample_texts):
        """测试批量文本识别"""
        mock_model.predict.return_value = [
            [{"entity": "ORG", "word": "UC Berkeley", "score": 0.95, "start": 0, "end": 11}],
            [{"entity": "ORG", "word": "Pfizer", "score": 0.92, "start": 0, "end": 6}],
            [{"entity": "PER", "word": "Albert Einstein", "score": 0.98, "start": 0, "end": 15}],
        ]
        
        response = client.post("/ner", json={"texts": sample_texts})
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3
    
    def test_ner_empty_request_rejected(self, client):
        """测试空请求被拒绝"""
        response = client.post("/ner", json={"texts": []})
        assert response.status_code == 422  # Validation error
    
    def test_ner_too_many_texts_rejected(self, client):
        """测试超过100条文本被拒绝"""
        texts = ["test text"] * 101
        response = client.post("/ner", json={"texts": texts})
        assert response.status_code == 400
        assert "Too many texts" in response.json()["detail"]
    
    def test_ner_invalid_request_format(self, client):
        """测试无效请求格式"""
        response = client.post("/ner", json={"invalid": "data"})
        assert response.status_code == 422
    
    def test_ner_response_entity_format(self, client, mock_model):
        """测试响应实体格式正确"""
        mock_model.predict.return_value = [[
            {"entity": "PER", "word": "Einstein", "score": 0.98, "start": 0, "end": 8}
        ]]
        
        response = client.post("/ner", json={"texts": ["Einstein was a physicist."]})
        data = response.json()
        
        entity = data["data"][0][0]
        assert "entity" in entity
        assert "word" in entity
        assert "score" in entity
        assert "start" in entity
        assert "end" in entity
        assert isinstance(entity["score"], float)
        assert isinstance(entity["start"], int)
        assert isinstance(entity["end"], int)


class TestErrorHandling:
    """错误处理测试"""
    
    def test_model_not_loaded_error(self, client, mock_model):
        """测试模型未加载时的错误处理"""
        mock_model.predict.side_effect = RuntimeError("Model not loaded")
        
        response = client.post("/ner", json={"texts": ["test"]})
        assert response.status_code == 503
    
    def test_internal_error_handling(self, client, mock_model):
        """测试内部错误处理"""
        mock_model.predict.side_effect = Exception("Unexpected error")
        
        response = client.post("/ner", json={"texts": ["test"]})
        assert response.status_code == 500
