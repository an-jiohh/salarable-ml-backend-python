from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.services.question_service import QuestionService, get_question_service

router = APIRouter()

# Request/Response Models
class CreateQuestionRequest(BaseModel):
    portfolio_data: str
    job_description_data: str
    input_position: str


class CreateQuestionResponse(BaseModel):
    output_data: dict

@router.post("/create_question", response_model=CreateQuestionResponse)
def query_search(request_model: CreateQuestionRequest, question_service: QuestionService = Depends(get_question_service)):
    logger = logging.getLogger(__name__)
    result = question_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
    logger.info(result)

    return CreateQuestionResponse(
            output_data=result,
        )