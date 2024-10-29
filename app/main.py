import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from app.routers import question_router, portfolio_router
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG
import asgi_correlation_id
import sentry_sdk
import os


def configure_logging():
    console_handler = logging.StreamHandler()
    console_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    # 파일 핸들러 (파일에 로그 기록)
    log_file = "/var/log/fastapi/app.log"   # 원하는 로그 파일 경로로 변경 가능
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7
    )
    file_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level="INFO",
        format="%(levelname)s %(asctime)s log [%(correlation_id)s] %(name)s %(message)s")


environment = os.getenv("ENVIRONMENT", "development")
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

#CORS
origins = [
    "https://gridge.salarable.pro",
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

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}