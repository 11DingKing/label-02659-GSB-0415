"""
Pydantic模型验证测试
"""
import pytest
from pydantic import ValidationError


class TestNERRequest:
    """NER请求模型测试"""
    
    def test_valid_request(self):
        """测试有效请求"""
        from app.schemas import NERRequest
        request = NERRequest(texts=["Hello world"])
        assert request.texts == ["Hello world"]
    
    def test_multiple_texts(self):
        """测试多条文本"""
        from app.schemas import NERRequest
        texts = ["Text 1", "Text 2", "Text 3"]
        request = NERRequest(texts=texts)
        assert len(request.texts) == 3
    
    def test_empty_texts_rejected(self):
        """测试空列表被拒绝"""
        from app.schemas import NERRequest
        with pytest.raises(ValidationError):
            NERRequest(texts=[])
    
    def test_max_texts_limit(self):
        """测试最大文本数量限制"""
        from app.schemas import NERRequest
        texts = ["text"] * 100
        request = NERRequest(texts=texts)
        assert len(request.texts) == 100
        
        with pytest.raises(ValidationError):
            NERRequest(texts=["text"] * 101)
    
    def test_missing_texts_field(self):
        """测试缺少texts字段"""
        from app.schemas import NERRequest
        with pytest.raises(ValidationError):
            NERRequest()


class TestNERResponse:
    """NER响应模型测试"""
    
    def test_valid_response(self):
        """测试有效响应"""
        from app.schemas import NERResponse
        response = NERResponse(
            success=True,
            data=[[]],
            message="OK"
        )
        assert response.success is True
    
    def test_response_with_entities(self):
        """测试包含实体的响应"""
        from app.schemas import NERResponse, Entity
        entity = Entity(entity="PER", word="John", score=0.95, start=0, end=4)
        response = NERResponse(
            success=True,
            data=[[entity]],
            message=None
        )
        assert len(response.data[0]) == 1
    
    def test_response_optional_message(self):
        """测试可选message字段"""
        from app.schemas import NERResponse
        response = NERResponse(success=True, data=[[]])
        assert response.message is None


class TestEntity:
    """实体模型测试"""
    
    def test_valid_entity(self):
        """测试有效实体"""
        from app.schemas import Entity
        entity = Entity(
            entity="PER",
            word="Einstein",
            score=0.98,
            start=0,
            end=8
        )
        assert entity.entity == "PER"
        assert entity.word == "Einstein"
        assert entity.score == 0.98
    
    def test_entity_missing_fields(self):
        """测试缺少必填字段"""
        from app.schemas import Entity
        with pytest.raises(ValidationError):
            Entity(entity="PER", word="John")  # 缺少score, start, end


class TestHealthResponse:
    """健康检查响应测试"""
    
    def test_valid_health_response(self):
        """测试有效健康响应"""
        from app.schemas import HealthResponse
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True
