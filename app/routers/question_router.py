from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.services.question_service import QuestionService, get_question_service
from app.services.question_service_v4 import QuestionServiceV4, get_question_service_v4
from app.services.question_service_v7 import QuestionServiceV7, get_question_service_v7
import traceback
import asyncio
from typing import Optional
from app.core.event_manager import add_event

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

class CreateQuestionResponseV7(BaseModel):
    main_questions_weakpoint_requirements:Optional[dict] = {}
    main_questions_weakpoint_preferences:Optional[dict] = {}
    followed_questions_weakpoint_requirements:Optional[dict] = {}
    followed_questions_weakpoint_preferences:Optional[dict] = {}
    main_questions_checkpoint_requirements:Optional[dict] = {}
    main_questions_checkpoint_preferences:Optional[dict] = {}
    followed_questions_checkpoint_requirements:Optional[dict] = {}
    followed_questions_checkpoint_preferences:Optional[dict] = {}
    questions_wowpoint:Optional[dict] = {}
    questions_doubtpoint:Optional[dict] = {}
    keywords_compared_with_keywordlist_and_jd:Optional[dict] = {}

# @router.post("/create_question", response_model=CreateQuestionResponse)
# def query_search(request_model: CreateQuestionRequest, question_service: QuestionService = Depends(get_question_service)):
#     result = question_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
#     logger.info(result)

#     return CreateQuestionResponse(
#             output_data=result,
#         )

@router.post("/v4/create_question", response_model=CreateQuestionResponse)
async def create_question(request_model: CreateQuestionRequest, question_service: QuestionServiceV4 = Depends(get_question_service_v4)):
    try :
        logger.info(f"portfolio_data : {request_model.portfolio_data}")
        logger.info(f"job_description_data : {request_model.job_description_data}")
        logger.info(f"input_position : {request_model.input_position}")
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

@router.post("/v7/create_question", response_model=CreateQuestionResponseV7)
async def create_question(request_model: CreateQuestionRequest, question_service: QuestionServiceV7 = Depends(get_question_service_v7)):
    try:
        logger.info(f"portfolio_data : {request_model.portfolio_data}")
        logger.info(f"job_description_data : {request_model.job_description_data}")
        logger.info(f"input_position : {request_model.input_position}")
        
        unique_keywords = question_service.unique_keywords

        #병렬 처리 1
        pf_task = question_service.preprocessing_pf(request_model.portfolio_data)
        jd_task = question_service.preprocessing_jd(request_model.job_description_data)
        pf_original, jd_original = await asyncio.gather(pf_task, jd_task)
        logger.info(f"pf_original : {pf_original}")
        await add_event({"type" : "pf_original", "data": pf_original})
        logger.info(f"jd_original : {jd_original}")
        await add_event({"type" : "jd_original", "data": jd_original})

        keywords_compared_with_keywordlist_and_jd = await question_service.extract_keyword_from_jd(unique_keywords,jd_original)

        logger.info(f"keywords_compared_with_keywordlist_and_jd : {keywords_compared_with_keywordlist_and_jd}")
        #병렬 처리 2
        conformitypoint_task = question_service.extract_conformitypoint(pf_original, keywords_compared_with_keywordlist_and_jd, request_model.input_position)
        wowpoint_task = question_service.extract_wowpoint(pf_original, keywords_compared_with_keywordlist_and_jd)
        doubtpoint_pf_only_task = question_service.extract_doubtpoint_pf_only(pf_original)

        conformitypoint, wowpoint, doubtpoint_pf_only = await asyncio.gather(conformitypoint_task, wowpoint_task, doubtpoint_pf_only_task)

        logger.info(f"conformitypoint : {conformitypoint}")
        logger.info(f"wowpoint : {wowpoint}")
        logger.info(f"doubtpoint_pf_only : {doubtpoint_pf_only}")
        await add_event({"type" : "conformitypoint", "data": conformitypoint})
        await add_event({"type" : "wowpoint", "data": wowpoint})
        await add_event({"type" : "doubtpoint_pf_only", "data": doubtpoint_pf_only})

        # **Extract WeakPoint & CheckPoint From ConformityPoint**
        weak_requirements_keywords, weak_preferences_keywords = [], []
        check_requirements_keywords, check_preferences_keywords = [], []
        for content in conformitypoint["conformitypoint"]:
            if not content["is_keywords_in_PF"]:
                if content["requirements_and_preferences"] == "requirements":
                    weak_requirements_keywords.append(content["tech_keywords"])
                elif content["requirements_and_preferences"] == "preferences":
                    weak_preferences_keywords.append(content["tech_keywords"])
            else:
                if content["requirements_and_preferences"] == "requirements":
                    check_requirements_keywords.append(content["tech_keywords"])
                elif content["requirements_and_preferences"] == "preferences":
                    check_preferences_keywords.append(content["tech_keywords"])

        await add_event({"type" : "weak_requirements_keywords", "data": weak_requirements_keywords})
        await add_event({"type" : "weak_preferences_keywords", "data": weak_preferences_keywords})
        await add_event({"type" : "check_requirements_keywords", "data": check_requirements_keywords})
        await add_event({"type" : "check_preferences_keywords", "data": check_preferences_keywords})
        
        # **Semantic Search**
        # 자격요건
        requirements_in_pf_semantic_search = {}
        requirements_in_pf_wiki_search = {}
        requirements_not_in_pf_semantic_search = {}
        requirements_not_in_pf_wiki_search = {}

        # 우대사항
        preferences_in_pf_semantic_search = {}
        preferences_in_pf_wiki_search = {}
        preferences_not_in_pf_semantic_search = {}
        preferences_not_in_pf_wiki_search = {}


        for content in conformitypoint["conformitypoint"]:
            keyword = content["tech_keywords"]
            _type = content["requirements_and_preferences"]
            is_in_pf = content["is_keywords_in_PF"]
            is_in_keyword_list = content["is_existed_in_keywordlist"]
            original_sentence = content["sentences"]
            temp_searched_sentences = []
        
            # 자격요건
            if _type == "requirements":
                # pf에 존재 O
                if is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        for s in original_sentence:
                            searched_data = question_service.search_vector_db(keyword, s, request_model.input_position)
                            question_service.append_semantic_search_result(
                                requirements_in_pf_semantic_search,
                                keyword,
                                s,
                                searched_data,
                            )
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass
                # pf에 존재 X
                elif not is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        searched_data = question_service.search_vector_db(keyword, keyword, request_model.input_position)
                        question_service.append_semantic_search_result(
                            requirements_not_in_pf_semantic_search,
                            keyword,
                            keyword,
                            searched_data,
                        )
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass

            # 우대사항
            elif _type == "preferences":
                # pf에 존재 O
                if is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        for s in original_sentence:
                            searched_data = question_service.search_vector_db(keyword, s, request_model.input_position)
                            question_service.append_semantic_search_result(
                                preferences_in_pf_semantic_search, keyword, s, searched_data
                            )
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass
                # pf에 존재 X
                elif not is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        searched_data = question_service.search_vector_db(keyword, keyword, request_model.input_position)
                        question_service.append_semantic_search_result(
                            preferences_not_in_pf_semantic_search,
                            keyword,
                            keyword,
                            searched_data,
                        )
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass

        await add_event({"type" : "requirements_in_pf_semantic_search", "data": requirements_in_pf_semantic_search})
        await add_event({"type" : "requirements_not_in_pf_semantic_search", "data": requirements_not_in_pf_semantic_search})
        await add_event({"type" : "preferences_in_pf_semantic_search", "data": preferences_in_pf_semantic_search})
        await add_event({"type" : "preferences_not_in_pf_semantic_search", "data": preferences_not_in_pf_semantic_search})

        # 병렬 처리 3
        # Weak-Point
        main_questions_weakpoint_requirements_task = question_service.generate_main_questions_weakpoint_requirements(weak_requirements_keywords)
        main_questions_weakpoint_preferences_task = question_service.generate_main_questions_weakpoint_preferences(weak_preferences_keywords)
        followed_questions_weakpoint_requirements_task = question_service.generate_followed_questions_weakpoint_requirements(requirements_not_in_pf_semantic_search)
        followed_questions_weakpoint_preferences_task = question_service.generate_followed_questions_weakpoint_preferences(preferences_not_in_pf_semantic_search)
        # Check-Point"
        main_questions_checkpoint_requirements_task = question_service.generate_main_questions_checkpoint_requirements(pf_original, check_requirements_keywords)
        main_questions_checkpoint_preferences_task = question_service.generate_main_questions_checkpoint_preferences(pf_original, check_preferences_keywords)
        followed_questions_checkpoint_requirements_task = question_service.generate_followed_questions_checkpoint_requirements(requirements_in_pf_semantic_search)
        followed_questions_checkpoint_preferences_task = question_service.generate_followed_questions_checkpoint_preferences(preferences_in_pf_semantic_search)
        questions_wowpoint_task = question_service.generate_questions_wowpoint(wowpoint)
        questions_doubtpoint_task = question_service.generate_questions_doubtpoint(doubtpoint_pf_only, pf_original)
        main_questions_weakpoint_requirements, main_questions_weakpoint_preferences,followed_questions_weakpoint_requirements,followed_questions_weakpoint_preferences,main_questions_checkpoint_requirements, main_questions_checkpoint_preferences, followed_questions_checkpoint_requirements, followed_questions_checkpoint_preferences, questions_wowpoint, questions_doubtpoint = await asyncio.gather(main_questions_weakpoint_requirements_task, main_questions_weakpoint_preferences_task, followed_questions_weakpoint_requirements_task, followed_questions_weakpoint_preferences_task, main_questions_checkpoint_requirements_task, main_questions_checkpoint_preferences_task, followed_questions_checkpoint_requirements_task, followed_questions_checkpoint_preferences_task, questions_wowpoint_task, questions_doubtpoint_task)

        return CreateQuestionResponseV7(
            main_questions_weakpoint_requirements=main_questions_weakpoint_requirements,
            main_questions_weakpoint_preferences=main_questions_weakpoint_preferences,
            followed_questions_weakpoint_requirements=followed_questions_weakpoint_requirements,
            followed_questions_weakpoint_preferences=followed_questions_weakpoint_preferences,
            main_questions_checkpoint_requirements=main_questions_checkpoint_requirements,
            main_questions_checkpoint_preferences=main_questions_checkpoint_preferences,
            followed_questions_checkpoint_requirements=followed_questions_checkpoint_requirements,
            followed_questions_checkpoint_preferences=followed_questions_checkpoint_preferences,
            questions_wowpoint=questions_wowpoint,
            questions_doubtpoint=questions_doubtpoint,
            keywords_compared_with_keywordlist_and_jd=keywords_compared_with_keywordlist_and_jd
        )

    except Exception as e:
        logger.error("Exception occurred: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))