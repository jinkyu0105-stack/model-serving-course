
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="분석할 평가 문장")

    model_config = {
        "json_schema_extra": {
            "examples":[
                {
                    "text": "매우 좋아"
                }
            ]
        }
    }

class PredictResponse(BaseModel):
    success: bool = Field(..., description="요청 처리 성공 여부")
    label: str = Field(..., description="예측된 감정 (긍정/부정)")
    confidence: float = Field(..., ge=0, le=1, description="모델의 예측확신도")

    model_config = {
        "json_schema_extra": {
            "examples":[
                {
                    "success": True,
                    "label": "긍정",
                    "confidence": 0.9985
                }
            ]
        }
    }
