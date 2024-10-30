from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.services.question_service import QuestionService, get_question_service
from app.services.question_service_v4 import QuestionServiceV4, get_question_service_v4
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

# Request/Response Models
class CreateQuestionRequest(BaseModel):
    portfolio_data: str
    job_description_data: str
    input_position: str


class CreateQuestionResponse(BaseModel):
    questions_wowpoint:dict
    questions_doubtpoint:dict
    questions_requirements_in_pf_semantic_search:dict
    questions_preferences_in_pf_semantic_search:dict
    questions_requirements_not_in_pf_semantic_search:dict
    questions_preferences_not_in_pf_semantic_search:dict

@router.post("/create_question", response_model=CreateQuestionResponse)
def query_search(request_model: CreateQuestionRequest, question_service: QuestionService = Depends(get_question_service)):
    result = question_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
    logger.info(result)

    return CreateQuestionResponse(
            output_data=result,
        )

@router.post("/v4/create_question", response_model=CreateQuestionResponse)
async def create_question(request_model: CreateQuestionRequest, question_service: QuestionServiceV4 = Depends(get_question_service_v4)):
    try :
        result = await question_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
    except Exception as e:
        logger.error("Exception occurred: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return CreateQuestionResponse(
        questions_wowpoint=result.questions_wowpoint,
        questions_doubtpoint=result.questions_doubtpoint,
        questions_requirements_in_pf_semantic_search=result.questions_requirements_in_pf_semantic_search,
        questions_preferences_in_pf_semantic_search=result.questions_preferences_in_pf_semantic_search,
        questions_requirements_not_in_pf_semantic_search=result.questions_requirements_not_in_pf_semantic_search,
        questions_preferences_not_in_pf_semantic_search=result.questions_preferences_not_in_pf_semantic_search
    )