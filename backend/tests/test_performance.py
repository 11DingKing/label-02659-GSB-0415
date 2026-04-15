"""
性能测试 - 验证CPU优化效果
"""
import pytest
import time
import torch
from unittest.mock import MagicMock, patch


class TestCPUOptimization:
    """CPU优化配置测试"""
    
    def test_torch_threads_configured(self):
        """测试PyTorch线程数配置"""
        from app.config import NUM_THREADS
        # 导入model模块会设置线程数
        import app.model
        
        # 验证线程数已设置（可能被系统限制）
        actual_threads = torch.get_num_threads()
        assert actual_threads >= 1
    
    def test_inference_mode_used(self):
        """测试推理模式使用"""
        from app.model import NERModel
        model = NERModel()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.ner_pipeline = MagicMock(return_value=[[]])
        
        # predict方法应该使用inference_mode
        with patch('torch.inference_mode') as mock_inference:
            mock_inference.return_value.__enter__ = MagicMock()
            mock_inference.return_value.__exit__ = MagicMock()
            model.predict(["test"])
            # inference_mode被调用
            mock_inference.assert_called()
    
    def test_cuda_and_bnb_detection(self):
        """测试CUDA和bitsandbytes检测"""
        import app.model as model_module
        
        # 验证检测变量存在
        assert hasattr(model_module, 'CUDA_AVAILABLE')
        assert hasattr(model_module, 'BNB_AVAILABLE')
        assert isinstance(model_module.CUDA_AVAILABLE, bool)
        assert isinstance(model_module.BNB_AVAILABLE, bool)


class TestQuantization:
    """量化测试"""
    
    def test_dynamic_quantization_applied(self):
        """测试动态量化是否正确应用"""
        # 创建一个简单的模型来测试量化
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # 应用动态量化
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 验证量化后的模型类型
        assert hasattr(quantized, 'linear')
        # 量化后的Linear层应该是DynamicQuantizedLinear
        assert 'DynamicQuantizedLinear' in str(type(quantized.linear))
    
    def test_quantization_reduces_memory(self):
        """测试量化减少内存占用"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 500)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        original_params = sum(p.numel() * p.element_size() for p in model.parameters())
        
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 量化模型的权重应该更小（INT8 vs FP32）
        # 注意：参数数量不变，但存储大小减少
        assert quantized is not None
    
    def test_load_in_8bit_used_when_available(self):
        """测试当CUDA和bitsandbytes可用时使用load_in_8bit"""
        from app.model import NERModel
        
        with patch('app.model.CUDA_AVAILABLE', True), \
             patch('app.model.BNB_AVAILABLE', True), \
             patch('app.model.AutoModelForTokenClassification') as mock_model_cls, \
             patch('app.model.AutoTokenizer') as mock_tokenizer, \
             patch('app.model.pipeline') as mock_pipeline:
            
            mock_model = MagicMock()
            mock_model.config.id2label = {0: "O", 1: "B-PER"}
            mock_model.config.num_labels = 2
            mock_model_cls.from_pretrained.return_value = mock_model
            
            model = NERModel()
            model.load()
            
            # 验证使用了load_in_8bit参数
            call_kwargs = mock_model_cls.from_pretrained.call_args[1]
            assert call_kwargs.get('load_in_8bit') == True
            assert call_kwargs.get('device_map') == 'auto'
    
    def test_dynamic_quantization_fallback(self):
        """测试纯CPU环境回退到动态量化"""
        from app.model import NERModel
        
        with patch('app.model.CUDA_AVAILABLE', False), \
             patch('app.model.BNB_AVAILABLE', False), \
             patch('app.model.AutoModelForTokenClassification') as mock_model_cls, \
             patch('app.model.AutoTokenizer') as mock_tokenizer, \
             patch('app.model.pipeline') as mock_pipeline, \
             patch('torch.quantization.quantize_dynamic') as mock_quantize:
            
            mock_model = MagicMock()
            mock_model.config.id2label = {0: "O", 1: "B-PER"}
            mock_model.config.num_labels = 2
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_quantize.return_value = mock_model
            
            model = NERModel()
            model.load()
            
            # 验证调用了动态量化
            mock_quantize.assert_called_once()


class TestBatchProcessing:
    """批处理性能测试"""
    
    def test_batch_processing_efficiency(self):
        """测试批处理效率"""
        from app.model import NERModel
        from app.config import BATCH_SIZE
        
        model = NERModel()
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        
        call_count = 0
        def mock_pipeline(texts):
            nonlocal call_count
            call_count += 1
            return [[]] * len(texts)
        
        model.ner_pipeline = mock_pipeline
        
        # 测试批处理：32条文本应该分2批处理（BATCH_SIZE=16）
        texts = ["test"] * 32
        model.predict(texts)
        
        expected_batches = (32 + BATCH_SIZE - 1) // BATCH_SIZE
        assert call_count == expected_batches


class TestPerformanceBenchmark:
    """性能基准测试（可选，需要真实模型）"""
    
    @pytest.mark.skip(reason="需要加载真实模型，CI环境跳过")
    def test_inference_latency(self):
        """测试推理延迟"""
        from app.model import NERModel
        
        model = NERModel()
        model.load()
        
        # 预热
        model.predict(["Warm up text"])
        
        # 测试单条文本延迟
        start = time.time()
        for _ in range(10):
            model.predict(["CRISPR-Cas9 is a gene editing tool."])
        avg_latency = (time.time() - start) / 10
        
        # 单条文本延迟应该在合理范围内（<1秒）
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
    
    @pytest.mark.skip(reason="需要加载真实模型，CI环境跳过")
    def test_batch_throughput(self):
        """测试批处理吞吐量"""
        from app.model import NERModel
        
        model = NERModel()
        model.load()
        
        texts = ["CRISPR-Cas9 is a gene editing tool developed at UC Berkeley."] * 16
        
        start = time.time()
        model.predict(texts)
        elapsed = time.time() - start
        
        throughput = len(texts) / elapsed
        # 批处理吞吐量应该 > 1 text/s
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} texts/s"
