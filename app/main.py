
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Depends, HTTPException
from app.auth import verify_api_key
from app.schemas import PredictRequest, PredictResponse
from app.model_service import load_model, predict
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware

logger = setup_logger(__name__)

app = FastAPI(
    title="한국어 특화 감성 분석 판독기",
    description="한국어 문장을 읽고 긍정/부정 판단하는 API.",
    version="1.0",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="sentiment_inference"
)
model = None

@app.on_event("startup")
async def startup():
    global model
    logger.info("감성 분석 모델 로드 중")
    model = load_model()
    logger.info("모델 로드 완료")

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "loading...",
        "loaded": model is not None,
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_sentiment(
    request: PredictRequest,
    user: str = Depends(verify_api_key)
    ):
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            predict,
            model,
            request.text
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "label": result["label"],
        "confidence": result["confidence"]
    }
