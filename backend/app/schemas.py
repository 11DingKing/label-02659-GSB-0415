from pydantic import BaseModel, Field
from typing import List, Optional, Any

class NERRequest(BaseModel):
    texts: List[str] = Field(
        ..., 
        description="待识别的文本列表（最多100条，单条最长2048字符）", 
        min_length=1,
        max_length=100
    )

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ..., 
        description="待识别的文本列表（最多100条，单条最长2048字符）", 
        min_length=1,
        max_length=100
    )
    threshold: Optional[float] = Field(
        None, 
        description="置信度阈值（0-1之间），过滤掉低于该分数的实体",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "CRISPR-Cas9 is a powerful gene editing tool developed at UC Berkeley.",
                    "The COVID-19 vaccine was developed by Pfizer and BioNTech."
                ]
            }
        }

class Entity(BaseModel):
    entity: str = Field(..., description="实体类型")
    word: str = Field(..., description="实体文本")
    score: float = Field(..., description="置信度分数")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")

class NERResponse(BaseModel):
    success: bool = Field(..., description="请求是否成功")
    data: List[List[Entity]] = Field(..., description="识别结果")
    message: Optional[str] = Field(None, description="附加信息")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
    model_config = {"protected_namespaces": ()}

class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="用户友好的错误描述")
    details: Optional[Any] = Field(None, description="额外错误详情")
