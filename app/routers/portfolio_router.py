from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.portfolio_service import PortfolioService, get_portfolio_service
import os
import logging
from typing import List


logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models

"""
# http제약사항으로 인해 파일 업로드를 위해 json이 아닌 Form형식으로 request
class MaskingPortfolioRequest(BaseModel):
    masking_text: str
    replacement_text: str
"""

class MaskingPortfolioResponse(BaseModel):
    output_data: dict

@router.post("/mask", response_model=MaskingPortfolioResponse)
async def masking_portfolio(
    background_tasks: BackgroundTasks,
    masking_text: List[str] = Form(...),  # 폼 데이터로 마스킹 텍스트 처리
    replacement_text: List[str] = Form(...),  # 폼 데이터로 대체 텍스트 처리
    file: UploadFile = File(...),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)):

    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(file_location)
        logger.info(masking_text)
        logger.info(replacement_text)
        
        # 무언가 포트폴리오 마스킹하는 로직
        # file_location을 service로 떨궈서 불러와서 사용
        # result = portfolio_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)

        background_tasks.add_task(delete_file, f"temp_{file.filename}")

        return FileResponse(path=f"temp_{file.filename}", filename=os.path.basename(f"temp_{file.filename}"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)