"""
NER模型单元测试
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch


class TestNERModelInit:
    """模型初始化测试"""
    
    def test_model_initial_state(self):
        """测试模型初始状态"""
        from app.model import NERModel
        model = NERModel()
        
        assert model.model is None
        assert model.tokenizer is None
        assert model.ner_pipeline is None
        assert model.label_map is None


class TestNERModelPredict:
    """模型预测功能测试"""
    
    @pytest.fixture
    def loaded_model(self):
        """创建已加载的模型实例"""
        from app.model import NERModel
        model = NERModel()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.ner_pipeline = MagicMock()
        model.label_map = {0: "O", 1: "B-PER", 2: "I-PER"}
        return model
    
    def test_predict_empty_list(self, loaded_model):
        """测试空列表输入"""
        result = loaded_model.predict([])
        assert result == []
    
    def test_predict_single_text(self, loaded_model):
        """测试单条文本预测"""
        loaded_model.ner_pipeline.return_value = [
            {"entity_group": "PER", "word": "Einstein", "score": 0.98, "start": 0, "end": 8}
        ]
        
        result = loaded_model.predict(["Einstein was a physicist."])
        
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["entity"] == "PER"
        assert result[0][0]["word"] == "Einstein"
    
    def test_predict_batch_texts(self, loaded_model):
        """测试批量文本预测"""
        loaded_model.ner_pipeline.return_value = [
            [{"entity_group": "PER", "word": "Einstein", "score": 0.98, "start": 0, "end": 8}],
            [{"entity_group": "ORG", "word": "MIT", "score": 0.95, "start": 0, "end": 3}],
        ]
        
        result = loaded_model.predict(["Einstein", "MIT"])
        
        assert len(result) == 2
    
    def test_predict_no_entities(self, loaded_model):
        """测试无实体文本"""
        loaded_model.ner_pipeline.return_value = [[]]
        
        result = loaded_model.predict(["Hello world"])
        
        assert len(result) == 1
        assert result[0] == []
    
    def test_predict_model_not_loaded(self):
        """测试模型未加载时抛出异常"""
        from app.model import NERModel
        model = NERModel()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict(["test"])
    
    def test_predict_long_text_truncation(self, loaded_model):
        """测试长文本截断"""
        loaded_model.ner_pipeline.return_value = [[]]
        long_text = "A" * 10000
        
        # 应该不抛出异常，内部会截断
        result = loaded_model.predict([long_text])
        assert len(result) == 1


class TestNERModelFormatResults:
    """结果格式化测试"""
    
    @pytest.fixture
    def model(self):
        from app.model import NERModel
        return NERModel()
    
    def test_format_empty_results(self, model):
        """测试空结果格式化"""
        result = model._format_results([])
        assert result == []
    
    def test_format_with_entity_group(self, model):
        """测试entity_group字段格式化"""
        input_data = [[
            {"entity_group": "PER", "word": "John", "score": 0.95, "start": 0, "end": 4}
        ]]
        result = model._format_results(input_data)
        
        assert result[0][0]["entity"] == "PER"
    
    def test_format_with_entity_field(self, model):
        """测试entity字段格式化（fallback）"""
        input_data = [[
            {"entity": "B-PER", "word": "John", "score": 0.95, "start": 0, "end": 4}
        ]]
        result = model._format_results(input_data)
        
        assert result[0][0]["entity"] == "B-PER"
    
    def test_format_score_rounding(self, model):
        """测试分数四舍五入"""
        input_data = [[
            {"entity_group": "PER", "word": "John", "score": 0.956789, "start": 0, "end": 4}
        ]]
        result = model._format_results(input_data)
        
        assert result[0][0]["score"] == 0.9568


class TestLabelMap:
    """标签映射测试"""
    
    def test_get_label_map_from_config(self):
        """测试从模型配置获取标签映射"""
        from app.model import NERModel
        model = NERModel()
        
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        model.model = mock_model
        
        label_map = model._get_label_map()
        
        assert label_map == {0: "O", 1: "B-PER", 2: "I-PER"}
    
    def test_get_label_map_default(self):
        """测试默认标签映射"""
        from app.model import NERModel
        model = NERModel()
        
        mock_model = MagicMock()
        mock_model.config.id2label = None
        mock_model.config.num_labels = 5
        model.model = mock_model
        
        label_map = model._get_label_map()
        
        assert len(label_map) == 5
        assert label_map[0] == "O"
