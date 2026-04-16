import logging
import time
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional
from .config import MODEL_NAME, NUM_THREADS, BATCH_SIZE, MAX_LENGTH

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CPU多线程优化 - 按照Prompt要求设置线程数
torch.set_num_threads(NUM_THREADS)
logger.info(f"PyTorch CPU threads set to: {NUM_THREADS}")

# 检测CUDA和bitsandbytes可用性
CUDA_AVAILABLE = torch.cuda.is_available()
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    logger.info("bitsandbytes available for INT8 quantization")
except ImportError:
    logger.info("bitsandbytes not available, will use torch dynamic quantization")


class NERModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self.label_map: Optional[Dict[int, str]] = None
        
    def load(self):
        """加载模型和tokenizer，支持CPU优化（参考Prompt示例代码）"""
        logger.info("=" * 50)
        logger.info("Starting model loading...")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Config: threads={NUM_THREADS}, batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}")
        logger.info(f"Environment: CUDA={CUDA_AVAILABLE}, bitsandbytes={BNB_AVAILABLE}")
        
        start_time = time.time()
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
        
        # CPU优化策略（按照Prompt示例代码）：
        # 1. 优先使用 load_in_8bit (bitsandbytes INT8量化) - 需要CUDA+bitsandbytes
        # 2. 回退到 torch.quantization.quantize_dynamic - 纯CPU环境
        
        if CUDA_AVAILABLE and BNB_AVAILABLE:
            # 按照Prompt示例：使用bitsandbytes INT8量化
            try:
                logger.info("Loading model with bitsandbytes INT8 quantization...")
                self.model = AutoModelForTokenClassification.from_pretrained(
                    MODEL_NAME,
                    load_in_8bit=True,
                    device_map="auto"
                )
                logger.info("Model loaded with load_in_8bit=True")
                # 创建NER pipeline (GPU)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=0
                )
            except Exception as e:
                logger.warning(f"load_in_8bit failed: {e}")
                logger.info("Falling back to dynamic quantization...")
                self._load_with_dynamic_quantization()
        else:
            # 纯CPU环境：使用PyTorch动态量化
            logger.info("Using CPU-only mode with dynamic quantization")
            self._load_with_dynamic_quantization()
        
        # 动态获取标签映射
        self.label_map = self._get_label_map()
        logger.info(f"Label map ({len(self.label_map)} labels): {self.label_map}")
        
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Model loading completed in {load_time:.2f}s")
        logger.info("=" * 50)
    
    def _load_with_dynamic_quantization(self):
        """使用PyTorch动态量化加载模型（纯CPU环境回退方案）"""
        logger.info("Loading base model...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )
        logger.info(f"Base model loaded: {self.model.config.num_labels} labels")
        
        # 应用CPU动态量化优化 - 减少内存占用并加速推理
        try:
            logger.info("Applying dynamic quantization (INT8)...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # 量化Linear层
                dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully")
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
            logger.info("Using standard FP32 model")
        
        # 创建NER pipeline (CPU)
        logger.info("Creating NER pipeline (CPU mode)...")
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=-1
        )
    
    def _get_label_map(self) -> Dict[int, str]:
        """从模型配置中动态获取标签映射"""
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            return self.model.config.id2label
        
        # 如果模型没有id2label，使用默认的BIO标签
        if hasattr(self.model.config, 'num_labels'):
            num_labels = self.model.config.num_labels
            default_labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", 
                            "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
            return {i: default_labels[i] if i < len(default_labels) else f"LABEL_{i}" 
                    for i in range(num_labels)}
        
        return {0: "O"}
    
    def predict(self, texts: List[str], threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """批量预测NER
        Args:
            texts: 待识别的文本列表
            threshold: 置信度阈值，低于该值的实体将被过滤
        """
        if not self.model:
            logger.error("Prediction failed: model not loaded")
            raise RuntimeError("Model not loaded")
        
        if not texts:
            logger.debug("Empty input, returning empty result")
            return []
        
        logger.debug(f"Processing {len(texts)} texts, threshold={threshold}")
        
        # 输入校验：截断过长文本
        processed_texts = []
        truncated_count = 0
        for idx, text in enumerate(texts):
            if len(text) > MAX_LENGTH * 4:  # 粗略估计字符数
                logger.warning(f"Text[{idx}] truncated: {len(text)} -> {MAX_LENGTH * 4} chars")
                text = text[:MAX_LENGTH * 4]
                truncated_count += 1
            processed_texts.append(text)
        
        if truncated_count > 0:
            logger.info(f"{truncated_count} texts were truncated due to length limit")
        
        results = []
        batch_count = (len(processed_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # 批量处理
        for batch_idx, i in enumerate(range(0, len(processed_texts), BATCH_SIZE)):
            batch = processed_texts[i:i + BATCH_SIZE]
            logger.debug(f"Processing batch {batch_idx + 1}/{batch_count} ({len(batch)} texts)")
            
            with torch.inference_mode():
                batch_results = self.ner_pipeline(batch)
                
                # 统一处理返回格式
                if len(batch) == 1:
                    if batch_results and isinstance(batch_results[0], dict):
                        batch_results = [batch_results]
                    elif not batch_results:
                        batch_results = [[]]
                else:
                    batch_results = [
                        r if isinstance(r, list) else [r] if r else []
                        for r in batch_results
                    ]
                
                results.extend(batch_results)
        
        return self._format_results(results, threshold)
    
    def _format_results(self, results: List, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """格式化输出结果
        Args:
            results: 模型返回的原始结果
            threshold: 置信度阈值，低于该值的实体将被过滤
        """
        formatted = []
        filtered_count = 0
        for doc_entities in results:
            entities = []
            if doc_entities:
                for ent in doc_entities:
                    if isinstance(ent, dict):
                        score = round(float(ent.get("score", 0)), 4)
                        if score >= threshold:
                            entities.append({
                                "entity": ent.get("entity_group", ent.get("entity", "UNKNOWN")),
                                "word": ent.get("word", ""),
                                "score": score,
                                "start": ent.get("start", 0),
                                "end": ent.get("end", 0)
                            })
                        else:
                            filtered_count += 1
            formatted.append(entities)
        
        if filtered_count > 0:
            logger.debug(f"Filtered out {filtered_count} entities with score < {threshold}")
        return formatted


# 全局模型实例
ner_model = NERModel()
