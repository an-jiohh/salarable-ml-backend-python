from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.portfolio_service import PortfolioService, get_portfolio_service
from app.utils.s3_uploader import S3Uploader, get_s3_uploader
import os
import logging
from typing import List
import boto3
from app.core.config import get_settings


#s3
config = get_settings()

s3_client = boto3.client('s3', 
                         aws_access_key_id = config.aws_access_key_id, 
                         aws_secret_access_key=config.aws_secret_access_key)

#logging
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

class MaskedTextRequest(BaseModel):
    file_name: str

class MaskedTextResponse(BaseModel):
    status: int
    masked_text: str



@router.post("/mask", response_model=MaskingPortfolioResponse)
async def masking_portfolio(
    background_tasks: BackgroundTasks,
    masking_text: List[str] = Form(...),  # 폼 데이터로 마스킹 텍스트 처리
    replacement_text: List[str] = Form(...),  # 폼 데이터로 대체 텍스트 처리
    file: UploadFile = File(...),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    s3_uploader: S3Uploader = Depends(get_s3_uploader)):

    try:

        masking_text.pop(0)
        replacement_text.pop(0)
    
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(file_location)
        logger.info(masking_text)
        logger.info(replacement_text)
        
        # 무언가 포트폴리오 마스킹하는 로직
        # file_location을 service로 떨궈서 불러와서 사용
        # result = po.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
        result = portfolio_service.mask_portfolio(file_location, masking_text, replacement_text)

        file_name = os.path.splitext(file.filename)[0]
        text_file_name = f"temp/{file_name}.txt"
        s3_uploader.upload_file(file_location, f"{file_name}/{file.filename}")
        s3_uploader.upload_file(result, f"{file_name}/masked_{file.filename}")
        s3_uploader.upload_file(text_file_name, f"{file_name}/{file_name}.txt")

        background_tasks.add_task(delete_file, file_location)
        background_tasks.add_task(delete_file, result)
        background_tasks.add_task(delete_file, text_file_name)

        return FileResponse(path=result, filename=os.path.basename(result))
    
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/maskedText", response_model=MaskedTextResponse)
async def get_masked_text(
    background_tasks: BackgroundTasks,
    request_model: MaskedTextRequest,
    s3_uploader: S3Uploader = Depends(get_s3_uploader)) :
    try :
        file_name = request_model.file_name
        s3_postion_name = os.path.splitext(file_name)[0]
        s3_uploader.download_file(f"temp/{s3_postion_name}.txt",f"{s3_postion_name}/{s3_postion_name}.txt" )

        with open(f"temp/{s3_postion_name}.txt", 'r', encoding='utf-8') as file:
            masked_text = file.read()

        background_tasks.add_task(delete_file, f"temp/{s3_postion_name}.txt")

        return MaskedTextResponse(
            status=200,
            masked_text=masked_text
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)