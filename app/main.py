import logging
from fastapi import FastAPI
from app.routers import question_router

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# 라우터 추가
app.include_router(question_router.router)

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}