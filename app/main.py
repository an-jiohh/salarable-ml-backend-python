import logging
from fastapi import FastAPI
from app.routers import question_router, portfolio_router
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import LOGGING_CONFIG
import asgi_correlation_id
import sentry_sdk


def configure_logging():
    console_handler = logging.StreamHandler()
    console_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    # 파일 핸들러 (파일에 로그 기록)
    log_file = "app.log"  # 원하는 로그 파일 경로로 변경 가능
    file_handler = logging.FileHandler(log_file)
    file_handler.addFilter(asgi_correlation_id.CorrelationIdFilter())

    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level="INFO",
        format="%(levelname)s %(asctime)s log [%(correlation_id)s] %(name)s %(message)s")


app = FastAPI(on_startup=[configure_logging])

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
    "http://localhost",
    "http://localhost:3000",
    "https://salarable.pro",
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