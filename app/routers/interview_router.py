from fastapi import FastAPI, WebSocket, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from pydantic import BaseModel
import uuid
import asyncio
from datetime import datetime
import json
import logging
from app.utils.RTZRClient import RTZRClient, get_rtzr_client
from app.services.interview_service import InterviewService, get_interview_service
from app.exceptions.interview import STTException
from app.services.question_service_v7 import QuestionServiceV7, get_question_service_v7
import traceback

logger = logging.getLogger(__name__)
router = APIRouter()

# Models
class InterviewStart(BaseModel):
    type: str
    timestamp: str
    user: str

class InterviewSession(BaseModel):
    id: str
    type: str
    start_time: str
    input_position: str
    evaluation_list: Dict
    questions: Dict
    main_questions_list: Dict
    current_question_index: int = 0
    is_active: bool = True
    current_question: Optional[str] = None
    main_quesions: List

# Technical interview questions
TECHNICAL_QUESTIONS = [
    "본인의 기술 스택에 대해 설명해주세요.",
    "가장 어려웠던 기술적 문제와 해결 방법은?",
    "RESTful API의 특징에 대해 설명해주세요.",
    "웹 성능 최적화 경험이 있다면 공유해주세요.",
]

# Behavioral interview questions
BEHAVIORAL_QUESTIONS = [
    "본인의 장단점에 대해 이야기해주세요.",
    "팀 프로젝트에서 갈등을 해결한 경험이 있나요?",
    "스트레스 상황에서 어떻게 대처하시나요?",
    "왜 우리 회사에 지원하셨나요?",
]


# 면접 세션 저장소
interview_sessions: Dict[str, InterviewSession] = {}

# 웹소켓 연결 저장소
active_connections: Dict[str, WebSocket] = {}

def get_questions_for_type(interview_type: str) -> List[str]:
    if interview_type == "technical":
        return TECHNICAL_QUESTIONS
    elif interview_type == "behavioral":
        return BEHAVIORAL_QUESTIONS
    else:  # comprehensive
        return TECHNICAL_QUESTIONS + BEHAVIORAL_QUESTIONS
    
class QuestionData(BaseModel):
    question: dict
    input_position: str
    
@router.post("/api/interviews/{interview_id}")
async def set_question(questions: QuestionData, interview_id: str):
    print(questions.question)
    session = interview_sessions.get(interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session.questions = questions.question
    session.input_position = questions.input_position
    for keyword in questions.question["keywords_compared_with_keywordlist_and_jd"].keys():
        if (questions.question["keywords_compared_with_keywordlist_and_jd"][keyword]["requirements_and_preferences"]
            == "requirements"
        ):
            session.evaluation_list["requirements"][keyword] = 0
        elif (
            questions.question["keywords_compared_with_keywordlist_and_jd"][keyword]["requirements_and_preferences"]
            == "preferences"
        ):
            session.evaluation_list["preferences"][keyword] = 0

    checkpoint_requirements = questions.question.get("main_questions_checkpoint_requirements")
    if checkpoint_requirements is not None:
        for content in checkpoint_requirements.get("requirements", []):
            session.main_questions_list["requirements"][content["tech_keyword"]] = content.get("question", "")
    
    weakpoint_requirements = questions.question.get("main_questions_weakpoint_requirements")
    if weakpoint_requirements is not None:
        for content in weakpoint_requirements.get("requirements", []):
            session.main_questions_list["requirements"][content["tech_keyword"]] = content.get("question", "")
    
    checkpoint_preferences = questions.question.get("main_questions_checkpoint_preferences")
    if checkpoint_preferences is not None:
        for content in checkpoint_preferences.get("preferences", []):
            session.main_questions_list["preferences"][content["tech_keyword"]] = content.get("question", "")
    
    weakpoint_preferences = questions.question.get("main_questions_weakpoint_preferences")
    if weakpoint_preferences is not None:
        for content in weakpoint_preferences.get("preferences", []):
            session.main_questions_list["preferences"][content["tech_keyword"]] = content.get("question", "")

    print(session)    

    return {"message": "Questions set successfully"}


@router.post("/api/interviews")
async def create_interview(interview_data: InterviewStart):
    interview_id = str(uuid.uuid4())
    questions = {}
    evaluation_list = {}
    evaluation_list["requirements"] = {}
    evaluation_list["preferences"] = {}

    main_questions_list = {}
    main_questions_list["requirements"] = {}
    main_questions_list["preferences"] = {}
    
    session = InterviewSession(
        id=interview_id,
        input_position="FE",
        type=interview_data.type,
        start_time=interview_data.timestamp,
        questions=questions,
        evaluation_list=evaluation_list,
        main_questions_list=main_questions_list,
        current_question="",
        main_quesions = []
    )
    
    interview_sessions[interview_id] = session
    return {"interviewId": interview_id}

@router.websocket("/ws/interview/{interview_id}")
async def interview_websocket(websocket: WebSocket, interview_id: str, stt_client: RTZRClient = Depends(get_rtzr_client), interview_service: InterviewService = Depends(get_interview_service), question_service: QuestionServiceV7 = Depends(get_question_service_v7)):
    await websocket.accept()
    active_connections[interview_id] = websocket
    count = 0
    try:
        session = interview_sessions.get(interview_id)
        if not session:
            await websocket.close(code=4000, reason="Invalid interview session")
            return
        types_count = 1
        score = 0
        audio_data = b""
        history = []
        main_question_flag = True
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        data = message["text"]
                        message = json.loads(data)
                        if message.get("type") == "start":
                            for _type in session.main_questions_list.keys():
                                for keyword, question in zip(
                                    session.main_questions_list[_type].keys(), session.main_questions_list[_type].values()
                                ):
                                    session.main_quesions.append({"type": _type, "keyword": keyword, "question": question, "depth":0})
                            print("types : ", session.main_quesions)
                            session.current_question = session.main_quesions[0]["question"]
                            await websocket.send_json({
                                "type": "question",
                                "text": session.main_quesions[0]["question"],
                                "keyword": session.main_quesions[0]["keyword"],
                                "questionNumber": session.current_question_index + 1,
                                "totalQuestions": len(session.questions)})
                        elif message.get("type") == "INTERVIEW_END":
                            await websocket.send_json({
                                "type": "interview_complete",
                                "message": "면접이 완료되었습니다.",
                                "score": score,
                                "history": history
                            })
                            break
                        elif message.get("type") == "END":
                            # 녹화 종료 메시지를 받으면 STT로 전송
                            if audio_data:
                                try :
                                    id = await stt_client.send_audio_file(audio_data)
                                    answer = await stt_client.poll_stt_status(id)
                                except Exception as e:
                                    print(f"Error in STT: {e}")
                                    answer = "STT에서 문제가 발생했어요. 다시 시도해주세요."
                                    raise STTException(e, 500)
                                evaluation = await question_service.evaluation_question(session.current_question, answer)
                                # next_question = await interview_service.get_quetions(answer)
                                if not main_question_flag:
                                    score += int(evaluation["score"])
                                history.append({"질문": session.current_question, "답변": answer, "평가": evaluation})
                                if count >= 3 or evaluation["score"] < 1:
                                    #{keyword}]에 대한 평가가 종료되었습니다. 평가점수는 {round((score / count), 2)}점 입니다."
                                    #다음대질문응로 넘김
                                    main_question_flag = True
                                    session.current_question_index += 1
                                    session.current_question = session.main_quesions[session.current_question_index]["question"]
                                else :
                                    count += 1
                                    searched_question, generated_question = await question_service.followed_question(evaluation, history, session.input_position)
                                    logger.info(f"searched_question : {searched_question}")
                                    logger.info(f"generated_question : {generated_question}")
                                    session.current_question = generated_question
                                print(f"STT Result: {answer}")
                                await websocket.send_json({
                                    "type": "question",
                                    "text": session.current_question,
                                    "questionNumber": session.current_question_index + 1,
                                    "totalQuestions": len(session.questions)
                                })
                                # with open(f"{interview_id}_audio.mp4", "wb") as f:
                                #     f.write(audio_data)
                                audio_data = b""  # 버퍼 초기화
                    elif "bytes" in message:
                        audio_data += message["bytes"]
                        print("Received data chunk:", len(message["bytes"]))
            except STTException as e:
                print(f"Error in STT: {e}")
                continue
            except Exception as e:
                print(f"Error in websocket connection: {e}")
                traceback.print_exc()
                break

            # message = json.loads(data)
            
            # if message.get("type") == "next_question":
            #     session.current_question_index += 1
            #     if session.current_question_index < len(session.questions):
            #         await websocket.send_json({
            #             "type": "question",
            #             "content": session.questions[session.current_question_index],
            #             "questionNumber": session.current_question_index + 1,
            #             "totalQuestions": len(session.questions)
            #         })
            #     else:
            #         await websocket.send_json({
            #             "type": "interview_complete",
            #             "message": "면접이 완료되었습니다."
            #         })
            #         break
            
            # elif message.get("type") == "end_interview":
            #     break

    except Exception as e:
        print(f"Error in websocket connection: {e}")
    
    finally:
        if interview_id in active_connections:
            del active_connections[interview_id]
        if interview_id in interview_sessions:
            interview_sessions[interview_id].is_active = False
        await websocket.close()

@router.get("/api/interviews/{interview_id}")
async def get_interview_status(interview_id: str):
    session = interview_sessions.get(interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    return {
        "id": session.id,
        "type": session.type,
        "currentQuestion": session.current_question_index + 1,
        "totalQuestions": len(session.questions),
        "isActive": session.is_active
    }