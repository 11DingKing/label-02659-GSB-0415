import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from .schemas import NERRequest, NERResponse, HealthResponse, ErrorResponse
from .model import ner_model
from .config import HOST, PORT, BATCH_SIZE

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 请求限制
MAX_TEXTS_PER_REQUEST = 100

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    logger.info("=" * 50)
    logger.info("Starting SciBERT NER Service...")
    try:
        ner_model.load()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    logger.info("=" * 50)
    yield
    logger.info("Shutting down SciBERT NER Service...")

app = FastAPI(
    title="SciBERT NER Service",
    description="基于SciBERT的专有名词识别服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({duration:.2f}ms)")
    
    return response

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        model_loaded=ner_model.model is not None
    )

@app.post("/ner", response_model=NERResponse, tags=["NER"])
async def predict_ner(request: NERRequest):
    """
    命名实体识别接口
    
    - **texts**: 待识别的文本列表（最多100条）
    """
    text_count = len(request.texts)
    total_chars = sum(len(t) for t in request.texts)
    logger.info(f"NER request: {text_count} texts, {total_chars} total chars")
    
    # 输入校验：限制批量请求数量
    if text_count > MAX_TEXTS_PER_REQUEST:
        logger.warning(f"Request rejected: {text_count} texts exceeds limit of {MAX_TEXTS_PER_REQUEST}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Too many texts",
                "message": f"请求包含 {text_count} 条文本，超过最大限制 {MAX_TEXTS_PER_REQUEST} 条",
                "max_allowed": MAX_TEXTS_PER_REQUEST,
                "received": text_count
            }
        )
    
    # 检查模型是否可用
    if ner_model.model is None:
        logger.error("NER request failed: model not loaded")
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Service unavailable",
                "message": "模型尚未加载完成，请稍后重试"
            }
        )
    
    try:
        start_time = time.time()
        results = ner_model.predict(request.texts)
        duration = (time.time() - start_time) * 1000
        
        entity_count = sum(len(entities) for entities in results)
        logger.info(f"NER completed: {entity_count} entities found in {duration:.2f}ms")
        
        return NERResponse(
            success=True,
            data=results,
            message=f"成功处理 {text_count} 条文本，识别出 {entity_count} 个实体"
        )
    except RuntimeError as e:
        logger.error(f"Model runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Model error",
                "message": "模型处理出错，请检查输入或稍后重试"
            }
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input",
                "message": f"输入数据格式错误: {str(e)}"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal server error",
                "message": "服务器内部错误，请联系管理员"
            }
        )

@app.get("/", tags=["Root"])
async def root():
    """API根路径"""
    return {
        "service": "SciBERT NER Service",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
