import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, Request
from app.routers import question_router, portfolio_router, interview_router, interviews_router
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG
import asgi_correlation_id
from asgi_correlation_id import correlation_id
import sentry_sdk
import os
from time import time

def configure_logging():
    console_handler = logging.StreamHandler()
    console_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    # 파일 핸들러 (파일에 로그 기록)
    log_file = os.getenv("LOG_PATH")  # 원하는 로그 파일 경로로 변경 가능
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7
    )
    file_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level="INFO",
        format="%(levelname)s %(asctime)s log [%(correlation_id)s] %(name)s %(message)s")


environment = os.getenv("ENVIRONMENT")
app = FastAPI(
    on_startup=[configure_logging],
    docs_url="/docs" if environment == "development" else None,
    redoc_url="/redoc" if environment == "development" else None
)

#centry init
sentry_sdk.init(
    dsn="https://6da3acf4369b05f894aff13dfae50290@o4508148534738944.ingest.us.sentry.io/4508148536049664",
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

#logging
app.add_middleware(asgi_correlation_id.CorrelationIdMiddleware)
@app.middleware("http")
async def log_request_response(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", "N/A")
    app_logger = logging.getLogger(__name__)

    start_time = time()

    ip_address = request.client.host
    forwarded_ip = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_ip.split(",")[0].strip() if forwarded_ip else ip_address
    
    # 요청 세부 정보를 로깅합니다.
    app_logger.info(f"Request: {request.method} {request.url} {client_ip} - Correlation ID: {correlation_id}")
    
    # 요청을 처리합니다.
    response = await call_next(request)
    
    # 소요 시간 계산
    duration = time() - start_time
    
    # 응답 세부 정보와 소요 시간을 로깅합니다.
    app_logger.info(f"Response: {response.status_code} - Correlation ID: {correlation_id} - Duration: {duration:.2f}s")
    
    return response

#CORS
origins = [
    "http://localhost:3000",
    "https://gridge.salarable.pro",
    "https://www.salarable.com",
    "https://salarable.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 추가
app.include_router(question_router.router)
app.include_router(portfolio_router.router, prefix="/portfolio")
app.include_router(interview_router.router)
app.include_router(interviews_router.router)

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}

from fastapi.responses import StreamingResponse
from app.core.event_manager import event_stream

# SSE 엔드포인트
@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(event_stream(), media_type="text/event-stream")
