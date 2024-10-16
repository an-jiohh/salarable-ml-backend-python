import logging
from fastapi import FastAPI
from app.routers import question_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

logging.basicConfig(level=logging.INFO)

#CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
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

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}