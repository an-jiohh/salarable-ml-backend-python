from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.services.question_service import QuestionService, get_question_service

router = APIRouter()

# Request/Response Models
class CreateQuestionRequest(BaseModel):
    state: str
    input_data: str

class CreateQuestionResponse(BaseModel):
    state: str
    output_data: str

@router.post("/create_question", response_model=CreateQuestionResponse)
def query_search(request_model: CreateQuestionRequest, question_service: QuestionService = Depends(get_question_service)):
    logger = logging.getLogger(__name__)
    result = question_service.create_questions()
    logger.info(result)

    return CreateQuestionResponse(
            state=request_model.state,
            output_data=request_model.input_data,
        )