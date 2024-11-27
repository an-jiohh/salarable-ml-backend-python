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
from app.core.event_manager import add_event

logger = logging.getLogger(__name__)
router = APIRouter()

# Models
class InterviewStart(BaseModel):
    timestamp: str
    user: str

class InterviewSession(BaseModel):
    id: str
    start_time: str
    input_position: str
    evaluation_list: Dict
    questions: Dict
    main_questions_list: Dict
    current_question_index: int = 0
    is_active: bool = True
    current_question: Optional[str] = None
    main_quesions: List
    history: List


# 면접 세션 저장소
interview_sessions: Dict[str, InterviewSession] = {}

# 웹소켓 연결 저장소
active_connections: Dict[str, WebSocket] = {}

class QuestionData(BaseModel):
    question: dict
    input_position: str


@router.post("/interviews")
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
        input_position="",
        start_time=interview_data.timestamp,
        questions=questions,
        evaluation_list=evaluation_list,
        main_questions_list=main_questions_list,
        current_question="",
        main_quesions = [],
        history=[]
    )
    
    interview_sessions[interview_id] = session
    return {"interviewId": interview_id}

@router.websocket("/interview/{interview_id}")
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

class CreateQuestionResponse(BaseModel):
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

pf_original = """
이력서 프론트엔드 개발자 마케터 시절 제품이 시장에 출시 된 후, 어떻게 자리 잡는지 직접 보고 싶어 웹개발을 시작했습니다. 3개의 사내 프로덕트를 메인으로 맡아, 섬세한 비즈니스 로직을 구현하기 위해 스스로 QA 항목을 만들어 팀원들과 공유 하며 데이터를 구축해왔습니다. 이 경험을 바탕으로, 섬세한 비즈니스 흐름을 제공하는 서비스를 만드는 데 기여하고 싶습니다. 또한, 서비스가 시장에서 어떤 반응을 얻고 있는지 빠르게 파악하고 적용하고자 합니다. 최근에는 앰플리튜드, 페이스북 픽셀과 같은 데이터 분석 및 마케팅 툴을 활용하여 추천 알고리즘을 개발하고, 이를 프론트엔드에서 어떻게 효과적으로 표현할지 고민하고 있습니다. 저는 말과 근거를 함께 제시하며 소통하는 것을 중요하게 생각합니다. 팀원들과 함께 좋은 서비스를 만들기 위해, 기술이 필요할 때는 문서와 데모를 제작하여 문제를 해결하는 방식으로 접근하고 있습니다.
\n
기술 스택
TypeScript, React, Next.js, tailwind-css, next-auth, storybook, react-query

경력
(주)브로즈 개발팀 | SW Developer | 정규직
2024. 02. ~ 재직 중
• 사내 신규 및 리뉴얼 서비스 4건 개발 참여, Tanstack-Query, React-hook-form 사내 도입
• (FAVIEW)-공간 뷰어 랜더링 서비스 개발
• 디자인 토큰 정리
• 디자인 토큰 별로 파일 정리, 기존의 파생되어있던 CSS property들을 정리된 디자인 시스템으로 모두 변경
• NaverMap API를 이용한, 자사 공간 데이터를 지도 서비스로 시각화
• CSR, React 18 환경에 맞춰, 기본 제공 로직 리팩토링 및 클러스터 내 마커(Marker)와 데이터 간 싱크 동기화.
• 주기적으로 ML 서버 호출을 담당하는 비즈니스 로직 리팩토링
• 기존: 기존 로직은 일정한 호출 스케줄링이 보장이 되지않아 의도와 다르게 호출이 되었음. 의존성 배열 미스매치로 과도한 리랜더링이 일어났음.
• 개선: useRef를 이용해 동일한 간격으로 함수 호출, Error, Done 상황에 따른 클로저 함수 작성 및 의존성 배열 재분리 및 과도한 리랜더링 방지
• React-Hook-form 도입
• 기존: Input등 인터렉션을 사용하는 페이지들이 많아 상태 관리가 과도하게 늘어나 있었음.
• 개선: control 메서드를 사용해 다양한 input, button, form, textarea 커스텀 컴포넌트로 분리, 코드 절반으로 감소. 작성 조건들을 객체로 관리하면서 각각의 페이지에서 컴포넌트를 사용할 때 일관성 유지.
• 이미지 최적화
• 기존: 유저가 공간 대표 사진을 업로드 시, 이미지 원본 그대로 업로드해서 이미지 랜더링이 느린 이슈가 있었음
• 개선: 이미지 리사이징 기준 논의 후, 업로드 시 리사이즈 및 최적화 후 서버로 전송
• 사내 홈페이지 리뉴얼
• 온보딩 페이지, 이벤트 페이지를 포함한 자사 소개 사이트 개발
• 공간 뷰어 랜더링 서비스 개발 레포지토리를 해당 서비스 레포지토리로 합치는 과정에서 TS 버전 차이로 인한 타입 오류 해결
• 이미지 최적화 및 속도 개선
• 로컬에서 이미지 자산의 확장자를 webp로 바꾸고, 기존 변경 전 파일 삭제 및 배럴파일의 확장자를 일괄적으로 바꿔주는 스크립트 작성
• 가장 무거운 익스텐션(파파고)이 설치된 환경에서 테스트 진행
• 라이트 하우스 기준 성능: 37 → 73, FCP: 3.7 → 0.7~1.2, LCP: 5.2 → 2.1- 3.4, JS절감: 약 116KiB 개선

(STORE) - 사내 커머스 서비스 웹뷰 작업 경험
• 기존의 타이포그래피 폰트는 고정값으로 설정되어있어, 작은 뷰포트 디바이스에서 기대하는 비율이 나오지 않음.
• 개선: 다양한 뷰포트에 대응하기 위해, 모바일 시안 너비에서의 px를 vw로 치환해, clamp()로 해결.
• IOS, AOS 실기기 테스트를 통해 네이티브 스타일에 따른 CSS 대응

(STORE ADMIN) - 사내 커머스 입점 점주를 위한 어드민 서비스 개발
• 현재 판매 상태를 나타내는 대시보드 페이지, 설정 페이지 개발
• 기존 사용하던, ContextAPI의 유저 action이 늘어나면서, 상태 관리의 어려움을 느낌.
• Tanstack-Query 도입 후 ContextAPI, 불필요한 상태 관리 제거
• 해당 쿼리가 업데이트가 되는 시점을 직관적인 코드로 표현해 이해를 도움

밸류와이즈 개발팀 | 프론트엔드 개발자 | 프리랜서
2023. 11. ~ 2023. 11. (1개월)
프로젝트 내용: 스톡 옵션 계산 서비스를 제공하는 프로덕트.
• next-auth와 AWS cognito 통합한 로그인 구현
• amazon-cognito-identity-js 라이브러리를 사용. 클라이언트에서 로그인 요청 시, 웹 로컬 스토리지에 5개의 provider과 사용자의 정보가 저장되는 문제 확인.
• 개선: next-auth의 credentials와 callback을 이용해 server-side 처리. 1개 쿠키로 정보를 관리.
• 결제 심사를 위한 페이지 개발 (공지사항, FAQ, 결제 페이지)
• 컴플라이언스& RFC 대응을 고려해 개발을 진행
• 서비스 운영 시, 사용자의 개인정보는 컴플레인이 많이 일어나는 요소라는 것을 알게 됨.
• 개선: 팀 내 데브옵스 엔지니어 분께 개인정보 위반 여부 컨펌 받은 후, RFC 이메일 패턴, 회원 가입 시, 체크 박스 전체 동의 조건 등 컴플라이언스를 가이드에 맞게 개발.

knccapitalManagement 개발팀 | 프론트엔드 개발자 | 프리랜서
프로젝트 내용: 뉴질랜드 투자 스타트업 회사 소개 페이지 리뉴얼.
• storybook을 이용한 소통
• UI 컴포넌트 단위 테스트 진행.
• Chromatic 배포를 통해 클라이언트가 레이아웃, 반응형 테스트를 직접 확인하고 피드백을 반영할 수 있는 컨트롤 인터페이스 제공.
• 반응형 UI 구현
• 복잡한 도형들을 가진 콘텐츠를 position과 가상요소 ::after를 이용해 직접 컴포넌트 제작.
• 클라이언트의 추상적인 요구와 느낌을 바탕으로 gsap을 이용해 마이크로 애니메이션 구현.
• 지역 인터넷 환경을 고려한 최적화
• 현지(뉴질랜드) 네트워크 상황을 개발 시에도 고려하기 위해 slow 3G, fast 3G 환경에서 랜더링 테스트 진행.
• 개선: slow 3G로 최초 진입 시, 평균 로드 시간이 3초 미만이 될 수 있도록 개발 진행. vite의 이미지 최적화를 이용. imagemin-cli 패키지를 이용해 로컬에서 한 번 더 이미지 최적화 진행.
• 웹 접근성을 고려한 개발 진행
• 시멘틱 태그를 사용.
• Firebase Cloud Functions를 이용한 메일 전송 기능 구현
• 기존 보유 서버가 node.js로 배포시 추가 요금 지불이 필요.
• 개선: express, nodemailer를 사용해 contact form에서 받은 정보를 회사 웹 메일로 전송하는 기능을 구현. 예민한 개인정보는 Magic key로 환경 변수 처리. firebase cloud functions를 선택해 배포 진행.

프로젝트 렛플(letspl) SF34(주식회사 에스에프써티포)
2024. 01. ~ 진행 중
• 퍼블리싱, 프론트엔드 개발 진행
• 회원가입 페이지 리뉴얼
• ui 라이브러리와 종속되어있어 무분별하게 증가한 !important제거. 기존 전역 css 파일 중 중복 되거나 덮여씌여진 CSS 2000줄 이상 제거. 들여쓰기로 사용되었던 문법들을 SCSS 함수를 이용해 클래스명을 동적으로 생성하는 방식으로 코드량 감사.
• 포인트 충전 페이지 UI 개발
• 튜토리얼 페이지 개발
• 유저 추천, 프로젝트 추천, 기능 가이드, 튜토리얼 등을 포함한 페이지 제작

포트폴리오 링크
벨로그 기술 블로그

교육
졸업 | 스포츠마케팅학과
2017. 03. ~ 2022. 02.

대외활동
프론트엔드 스쿨 플러스 단기 심화 과정 최종 발표 심사위원 (주)멋쟁이사자처럼 2023
• 정규 프론트엔드 과정을 수료하고, 심화 과정을 수강한 학생들의 파이널 프로젝트를 심사 및 피드백.
• 기획 의도에 맞는 UX 설계 수준, 완성도를 평가.
• 서비스 타입의 컴플라이언스 평가.
• 기본 동작 테스트, 테스트 케이스 평가.
• 코드 레벨 피드백.

cognito와 next-auth를 통합한 커스텀 로그인 페이지 만들기 발표
AWS 사용자 모임 - 프론트엔드 소모임 2024
• cognito-identity-js와 next-auth를 통합한 유연한 UI를 구축하는 경험을 공유.
• next-auth의 credentials을 이용한 서버사이드 인증 처리 과정 소개.

관련 레포지토리 링크

(주) 멋쟁이사자처럼 프론트엔드 스쿨 특강 강사 (주) 멋쟁이사자처럼 2023
1. MVP 설정을 통한 파이널 프로젝트 준비하기 (2023.08)
• 특강 내용: 2시간 특강 + 2시간 맨 투 맨 멘토링 진행.
• “3주의 시간동안 어떻게 효율적으로 프로젝트를 진행 할 것인가?”라는 관점에서 학생들이 많이 겪는 방해 요소를 정리.
• mvp 소개.
• IA & UserFlow, 백로그 등 개발에 앞서 제작해야 하는 과정에 대한 당위성 설명.

2. 개발 프로젝트 기획 방법 (2023.02)
• 특강 내용: 경험을 녹인 기획 프로세스에 대한 전반적인 설명.
• 도큐먼트 작성 방법에 대한 케이스 스터디.

3. github를 이용한 프로젝트 협업 방법 (2023.01)
• 특강 내용: 깃이슈, 마일스톤, 프로젝트 사용법 소개.
• 케이스 스터디를 통한 이해 고도화.
• 라이브러리의 깃 이슈와 디스커션을 활용한 정보 교환 사례 소개."""
jd_original = """
자격요건:
- WEB, HTML, CSS, JavaScript, TypeScript에 대한 높은 이해도 및 지식,
- Context API, Redux, Zustand, Tanstack-Query, Graphql Client 등을 활용한 앱 상태관리 경험,
- 프론트엔드 개발 경력 2년 이하
우대사항:
- Graphql을 사용하여 개발해본 경험 및 스키마 디자인에 대한 고민 경험,"""
keywords_compared_with_keywordlist_and_jd = """
{'html': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'css': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'javascript': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'typescript': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'context api': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'redux': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'zustand': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'tanstack query': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': True}, 'graphql': {'requirements_and_preferences': 'preferences', 'is_existed_in_keywordlist': False}, 'graphql client': {'requirements_and_preferences': 'requirements', 'is_existed_in_keywordlist': False}}
"""
conformitypoint = """tech_keywords': 'html'
'sentences': 
'ui 라이브러리와 종속되어있어 무분별하게 증가한 !important제거. 기존 전역 css 파일 중 중복 되거나 덮여씌여진 CSS 2000줄 이상 제거. 들여쓰기로 사용되었던 문법들을 SCSS 함수를 이용해 클래스명을 동적으로 생성하는 방식으로 코드량 감사.', 

'tech_keywords': 'css', 
'sentences': 
'ui 라이브러리와 종속되어있어 무분별하게 증가한 !important제거. 기존 전역 css 파일 중 중복 되거나 덮여씌여진 CSS 2000줄 이상 제거. 들여쓰기로 사용되었던 문법들을 SCSS 함수를 이용해 클래스명을 동적으로 생성하는 방식으로 코드량 감사.', 
'IOS, AOS 실기기 테스트를 통해 네이티브 스타일에 따른 CSS 대응', 

'tech_keywords': 'javascript', 
'sentences': 
'CSR, React 18 환경에 맞춰, 기본 제공 로직 리팩토링 및 클러스터 내 마커(Marker)와 데이터 간 싱크 동기화.', 
'기존: 기존 로직은 일정한 호출 스케줄링이 보장이 되지않아 의도와 다르게 호출이 되었음. 의존성 배열 미스매치로 과도한 리랜더링이 일어났음.', 
'개선: useRef를 이용해 동일한 간격으로 함수 호출, Error, Done 상황에 따른 클로저 함수 작성 및 의존성 배열 재분리 및 과도한 리랜더링 방지', 

'tech_keywords': 'typescript'
'sentences': 
'공간 뷰어 랜더링 서비스 개발 레포지토리를 해당 서비스 레포지토리로 합치는 과정에서 TS 버전 차이로 인한 타입 오류 해결',

'tech_keywords': 'context api'
'sentences': 
'기존 사용하던, ContextAPI의 유저 action이 늘어나면서, 상태 관리의 어려움을 느낌.',
'Tanstack-Query 도입 후 ContextAPI, 불필요한 상태 관리 제거'

'tech_keywords': 'tanstack query', 
'sentences': 
'사내 신규 및 리뉴얼 서비스 4건 개발 참여, Tanstack-Query, React-hook-form 사내 도입',
'Tanstack-Query 도입 후 ContextAPI, 불필요한 상태 관리 제거',

'tech_keywords': 'redux'
'sentences': ['redux']

'tech_keywords': 'zustand'
'sentences': ['zustand'], 
"""
wowpoint = """ 
'wowpoints': 
'3개의 사내 프로덕트를 메인으로 맡아, 섬세한 비즈니스 로직을 구현하기 위해 스스로 QA 항목을 만들어 팀원들과 공유하며 데이터를 구축해왔습니다.', 
'서비스가 시장에서 어떤 반응을 얻고 있는지 빠르게 파악하고 적용하고자 합니다.',
'말과 근거를 함께 제시하며 소통하는 것을 중요하게 생각합니다.'"""

doubtpoint_pf_only = """
'original_sentences': '졸업 | 스포츠마케팅학과 2017. 03. ~ 2022. 02.', 
'reason': '스포츠마케팅학과 졸업 후, 개발 관련 직군으로 빠르게 전환한 점이 의심스러울 수 있습니다.',

'original_sentences': '밸류와이즈 개발팀 | 프론트엔드 개발자 | 프리랜서 2023. 11. ~ 2023. 11. (1개월)', 
'reason': '프리랜서로 1개월간의 짧은 경력은 프로젝트의 깊이나 기여도를 의심하게 만들 수 있습니다.'"""
weak_requirements_keywords = "['redux', 'zustand', 'graphql client']"
weak_preferences_keywords = "['graphql']"
check_requirements_keywords = "['html', 'css', 'javascript', 'typescript', 'context api', 'tanstack query']"
check_preferences_keywords = "[]"
requirements_in_pf_semantic_search = """{"html": {"ui 라이브러리와 종속되어있어 무분별하게 증가한 !important제거. 기존 전역 css 파일 중 중복 되거나 덮여씌여진 CSS 2000줄 이상 제거. 들여쓰기로 사용되었던 문법들을 SCSS 함수를 이용해 클래스명을 동적으로 생성하는 방식으로 코드량 감사.": [{"searched_question": "프론트엔드 개발 시, React가 아닌 HTML, CSS, 바닐라 자바스크립트로만 프로젝트를 만들어본 경험이 있으신가요?", "score": 0.68}, {"searched_question": "HTML5에서 도입된 주요 태그들에 대해 설명해 주세요.", "score": 0.68}, {"searched_question": "<section>과 <article> 태그의 차이에 대해 설명해 주세요.", "score": 0.64}]}, "css": {"ui 라이브러리와 종속되어있어 무분별하게 증가한 !important제거. 기존 전역 css 파일 중 중복 되거나 덮여씌여진 CSS 2000줄 이상 제거. 들여쓰기로 사용되었던 문법들을 SCSS 함수를 이용해 클래스명을 동적으로 생성하는 방식으로 코드량 감사.": [{"searched_question": "CSS의 display 속성에 어떤 값들이 있는지 설명해 주세요. 아시는 속성들을 말씀해 주실 수 있나요?", "score": 0.73}, {"searched_question": "CSS의 flex 속성에 대해 설명해 주세요. flex 속성이 어떻게 구성되고, 각각의 값들이 어떤 역할을 하는지 설명해 주실 수 있나요?", "score": 0.71}, {"searched_question": "페이지에서 표준 폰트가 아닌 폰트 디자인을 사용할 때 어떤 방식으로 처리하는지 설명해 주세요.", "score": 0.7}], "IOS, AOS 실기기 테스트를 통해 네이티브 스타일에 따른 CSS 대응": [{"searched_question": "CSS에서 클리어링(Clearing)을 위한 다양한 방법과 각각을 언제 사용하는지 설명해 주세요.", "score": 0.76}, {"searched_question": "box model이 무엇이며, 브라우저에서 어떻게 동작하는지 설명해 주세요", "score": 0.75}, {"searched_question": "CSS 레이아웃과 관련된 margin, border, padding, content 속성들은 각각 어떤 영역을 차지하는지 설명해 주세요.", "score": 0.74}]}, "javascript": {"CSR, React 18 환경에 맞춰, 기본 제공 로직 리팩토링 및 클러스터 내 마커(Marker)와 데이터 간 싱크 동기화.": [{"searched_question": "자바스크립트의 Number Type은 다른 언어들과 차이점이 무엇인가, 왜 하나만 존재하나.", "score": 0.8}, {"searched_question": "JavaScript는 싱글 스레드 언어이지만 병렬처리가 가능한 이유에 대해 설명해주세요.", "score": 0.79}, {"searched_question": "\bVue.js가 다른 자바스크립트 기반 언어와 비교하여 어떤 장단점이 있는지 설명해주세요.", "score": 0.78}], "기존: 기존 로직은 일정한 호출 스케줄링이 보장이 되지않아 의도와 다르게 호출이 되었음. 의존성 배열 미스매치로 과도한 리랜더링이 일어났음.": [{"searched_question": "자바스크립트의 배열이 실제 자료구조 배열이 아닌 이유는 무엇인가요?", "score": 0.76}, {"searched_question": "이벤트 버블링이 기본적으로 child에서 parent로 전파되는데, 이를 반대로 구현하는 방법에 대해 설명해 주세요.", "score": 0.76}, {"searched_question": "자바스크립트의 Number Type은 다른 언어들과 차이점이 무엇인가, 왜 하나만 존재하나.", "score": 0.76}], "개선: useRef를 이용해 동일한 간격으로 함수 호출, Error, Done 상황에 따른 클로저 함수 작성 및 의존성 배열 재분리 및 과도한 리랜더링 방지": [{"searched_question": "자바스크립트의 Number Type은 다른 언어들과 차이점이 무엇인가, 왜 하나만 존재하나.", "score": 0.64}, {"searched_question": "javascript 의 트리쉐이킹에 대해서 설명하시고, 일어나는 조건과 일어나지 않는 조건에 대해 설명해주세요", "score": 0.63}, {"searched_question": "자바스크립트의 원시 타입은 몇 가지이며, 그 종류는 무엇인가요?", "score": 0.6}]}, "typescript": {"공간 뷰어 랜더링 서비스 개발 레포지토리를 해당 서비스 레포지토리로 합치는 과정에서 TS 버전 차이로 인한 타입 오류 해결": [{"searched_question": "Typescript 로 작성한 react 코드가 브라우저에서 실행가능한 코드로 변경되는 과정을 설명해주세요", "score": 0.83}, {"searched_question": "Typescript 의 타입으로 처리할 수 있는 동작과, Typescript 타입 정의로 불가능한 검증의 사례를 들어주세요", "score": 0.79}, {"searched_question": "프로젝트에서 사용한 Typescript 중 가장 구성하기 까다로웠던 type 에 대해 설명해주세요", "score": 0.79}]}, "context api": {"기존 사용하던, ContextAPI의 유저 action이 늘어나면서, 상태 관리의 어려움을 느낌.": [{"searched_question": "React의 상태 관리 도구(예: Redux, Context API)를 사용하는 이유와 장점은 무엇인가요?", "score": 0.74}, {"searched_question": "Context API에 대해 설명해 주실 수 있나요?", "score": 0.64}], "Tanstack-Query 도입 후 ContextAPI, 불필요한 상태 관리 제거": [{"searched_question": "React의 상태 관리 도구(예: Redux, Context API)를 사용하는 이유와 장점은 무엇인가요?", "score": 0.74}, {"searched_question": "Context API에 대해 설명해 주실 수 있나요?", "score": 0.68}]}, "tanstack query": {"사내 신규 및 리뉴얼 서비스 4건 개발 참여, Tanstack-Query, React-hook-form 사내 도입": [{"searched_question": "TanStack Query의 주요 특징과 사용 사례는 무엇인가요?", "score": 0.67}, {"searched_question": "TanStack Query를 사용하는 주된 이유는 무엇인가요?", "score": 0.66}], "Tanstack-Query 도입 후 ContextAPI, 불필요한 상태 관리 제거": [{"searched_question": "TanStack Query를 사용하는 주된 이유는 무엇인가요?", "score": 0.68}, {"searched_question": "TanStack Query의 주요 특징과 사용 사례는 무엇인가요?", "score": 0.68}]}}"""
requirements_not_in_pf_semantic_search = """{"redux": {"redux": [{"searched_question": "React의 상태 관리 도구(예: Redux, Context API)를 사용하는 이유와 장점은 무엇인가요?", "score": 0.62}, {"searched_question": "Redux의 상태 관리 패턴과 구조는 어떻게 되나요?", "score": 0.57}, {"searched_question": "React의 상태 관리에 대해 알고 있나요? Redux를 사용해 본 경험이 있다면, 그것에 대해 설명해 주실 수 있나요?", "score": 0.54}]}, "zustand": {"zustand": [{"searched_question": "Zustand를 사용하여 상태 관리를 어떻게 구현할 수 있나요?", "score": 0.56}, {"searched_question": "Zustand의 주요 특징과 사용 사례는 무엇인가요?", "score": 0.53}]}}"""
preferences_in_pf_semantic_search = "{}"
preferences_not_in_pf_semantic_search = "{}"

main_questions_weakpoint_requirements = """
'채용공고에서 redux를 요구하고 있는데, 포트폴리오/이력서에는 해당 내용이 존재하지 않습니다. 혹시 redux를 사용해보신 경험이나 이론적으로 알고 계신 내용이 있나요?
'tech_keyword': 'redux',
'question_type': '경험',
'purpose': '이 사용자가 redux를 사용해 봤는지에 대한 사실 확인',
'example': 
    "우수 : '네, redux를 사용하여 상태 관리를 구현한 경험이 있습니다. 특히 비동기 작업을 처리하기 위해 redux-thunk를 사용했습니다.',
    "보통 : 'redux에 대해 공부한 적은 있지만 실제 프로젝트에 적용해본 적은 없습니다.',
    "미흡 : 'redux가 무엇인지 잘 모릅니다.',
'reason': 'redux 사용 경험을 통해 상태 관리에 대한 이해도를 평가하기 위함입니다.',

'채용공고에서 zustand를 요구하고 있는데, 포트폴리오/이력서에는 해당 내용이 존재하지 않습니다. 혹시 zustand를 사용해보신 경험이나 이론적으로 알고 계신 내용이 있나요?',
'tech_keyword': 'zustand',
'question_type': '경험',
'purpose': '이 사용자가 zustand를 사용해 봤는지에 대한 사실 확인',
'example': 
    "우수 : '네, zustand를 사용하여 React 애플리케이션의 상태 관리를 간단하게 구현한 경험이 있습니다. 특히 작은 규모의 프로젝트에서 유용하게 사용했습니다.',
    "보통 : 'zustand에 대해 들어본 적은 있지만, 실제로 사용해본 적은 없습니다.',
    "미흡 : 'zustand가 무엇인지 잘 모릅니다.',
'reason': 'zustand 사용 경험을 통해 상태 관리에 대한 이해도를 평가하기 위함입니다.'
"""
main_questions_weakpoint_preferences = """
'채용공고에서 graphql를 요구하고 있는데, 포트폴리오/이력서에는 해당 내용이 존재하지 않습니다. 혹시 graphql를 사용해보신 경험이나 이론적으로 알고 계신 내용이 있나요?',
'tech_keyword': 'graphql',
'question_type': '경험',
'purpose': '이 사용자가 해당 기술을 사용해 봤는지에 대한 사실 확인',
'example': 
    "우수 : '네, 이전 프로젝트에서 GraphQL을 사용하여 API를 구축한 경험이 있습니다. 특히 데이터 페칭 최적화에 많은 도움이 되었습니다.',
    "보통 : '이론적으로는 알고 있지만, 실무에서 사용해본 적은 없습니다. 학습을 통해 빠르게 익힐 수 있습니다.'",
    "미흡 : 'GraphQL이 무엇인지 잘 모릅니다. REST API와 비슷한 건가요?'"],
'reason': '지원자가 GraphQL에 대한 실무 경험이나 이론적 이해를 가지고 있는지 확인하기 위함입니다.'
"""
followed_questions_weakpoint_requirements = """
tech_keyword : redux
Redux를 사용하여 상태 관리를 구현할 때의 주요 이점은 무엇인가요?
Redux의 미들웨어를 활용한 비동기 작업 처리는 어떻게 이루어지나요?
Redux의 액션과 리듀서의 역할과 관계에 대해 설명해 주세요.
tech_keyword : zustand
Zustand의 상태 관리 방식은 Redux와 어떻게 다른가요?
Zustand를 사용하여 상태를 공유할 때의 장점은 무엇인가요?
Zustand의 API를 사용하여 상태를 업데이트하는 방법을 설명해 주세요.
"""
followed_questions_weakpoint_preferences = """
tech_keyword : None
None
"""
main_questions_checkpoint_requirements = """
'TypeScript를 사용하여 프로젝트를 진행한 경험이 있으신가요? 예를 들어, 어떤 프로젝트에서 TypeScript를 활용하셨는지 설명해 주실 수 있나요?',
'tech_keyword': 'typescript',
'question_type': '경험',
'purpose': '이 사용자가 TypeScript를 실제 프로젝트에서 사용해 본 경험이 있는지 확인하기 위함입니다.',
'example': 
    "우수 : '네, TypeScript를 사용하여 (FAVIEW) 공간 뷰어 랜더링 서비스 개발 프로젝트에서 타입 안정성을 높이고, 코드의 가독성을 개선했습니다. 특히, TS 버전 차이로 인한 타입 오류를 해결한 경험이 있습니다.'",
    "보통 : 'TypeScript를 사용한 경험이 있습니다. 주로 타입 안정성을 위해 사용했습니다.'",
    "미흡 : 'TypeScript를 들어본 적은 있지만, 사용해 본 적은 없습니다.'",
'reason': '포트폴리오에 TypeScript 사용 경험이 명시되어 있으나, 구체적인 프로젝트나 상황이 언급되지 않았기 때문에 이를 확인하기 위한 질문입니다.',

'Tanstack Query를 도입하여 프로젝트를 개선한 경험이 있으신가요? 구체적으로 어떤 문제를 해결하기 위해 사용하셨는지 설명해 주실 수 있나요?',
'tech_keyword': 'tanstack query',
'question_type': '경험',
'purpose': '이 사용자가 Tanstack Query를 실제 프로젝트에서 사용해 본 경험이 있는지 확인하기 위함입니다.',
'example': 
    "우수 : '네, Tanstack Query를 도입하여 STORE ADMIN 프로젝트에서 ContextAPI의 복잡한 상태 관리를 개선했습니다. 이를 통해 불필요한 상태 관리를 제거하고, 쿼리 업데이트 시점을 직관적으로 표현할 수 있었습니다.'",
    "보통 : 'Tanstack Query를 사용하여 상태 관리를 개선한 경험이 있습니다.'",
    "미흡 : 'Tanstack Query에 대해 들어본 적은 있지만, 사용해 본 적은 없습니다.'",
'reason': '포트폴리오에 Tanstack Query 사용 경험이 명시되어 있으나, 구체적인 프로젝트나 상황이 언급되지 않았기 때문에 이를 확인하기 위한 질문입니다.'
"""
main_questions_checkpoint_preferences = """
'당신의 포트폴리오에 따르면, 다양한 뷰포트에 대응하기 위해 모바일 시안 너비에서의 px를 vw로 치환하고 clamp()를 사용하여 문제를 해결했다고 하셨습니다. 이 경험을 통해 얻은 교훈이나 개선된 점에 대해 설명해 주실 수 있나요?',
'tech_keyword': 'ui/ux',
'question_type': '경험',
'purpose': '이 사용자가 UI/UX 관련 기술을 실제로 사용해 본 경험이 있는지 확인하기 위함입니다.',
'example':
    '우수 : 다양한 뷰포트에 대응하기 위해 vw와 clamp()를 사용하여 반응형 디자인을 구현했습니다. 이를 통해 사용자 경험이 크게 개선되었고, 다양한 디바이스에서 일관된 UI를 제공할 수 있었습니다.',
    '보통 : 모바일에서의 UI 문제를 해결하기 위해 vw와 clamp()를 사용했습니다. 결과적으로 UI가 좀 더 유연해졌습니다.',
    '미흡 : vw와 clamp()를 사용해봤지만, 큰 차이를 느끼지 못했습니다.',
'reason': '포트폴리오에 UI/UX 관련 경험이 언급되어 있으나, 구체적인 기술 사용 경험을 확인하기 위해 질문을 생성했습니다.'
"""
followed_questions_checkpoint_requirements = """필수항목

tech_keyword': 'html',
'HTML5에서 추가된 시맨틱 태그들에 대해 설명해 주세요.',
'HTML 문서의 구조를 어떻게 설계하는지 설명해 주세요.',
'HTML에서 접근성을 고려한 마크업 방법에 대해 설명해 주세요.'

tech_keyword': 'css',
'CSS에서 Flexbox와 Grid의 차이점에 대해 설명해 주세요.',
'CSS의 Box Model이 무엇인지 설명하고, 각 구성 요소에 대해 설명해 주세요.',
'CSS에서 미디어 쿼리를 사용하여 반응형 디자인을 구현하는 방법에 대해 설명해 주세요.',

'tech_keyword': 'javascript',
'questions': ['자바스크립트의 비동기 처리 방식에 대해 설명해 주세요.',
'자바스크립트 클로저(Closure)에 대해 설명해 주세요.',
'자바스크립트에서 프로미스(Promise)와 async/await의 차이점에 대해 설명해 주세요.',

'tech_keyword': 'typescript',
'questions': ['TypeScript에서 인터페이스와 타입의 차이점에 대해 설명해 주세요.',
'TypeScript의 제네릭(Generic)에 대해 설명해 주세요.',
'TypeScript에서 유니온 타입과 인터섹션 타입의 차이점에 대해 설명해 주세요.',

'tech_keyword': 'context api',
'Context API를 사용하여 상태 관리를 할 때의 장단점에 대해 설명해 주세요.',
'Context API와 Redux의 차이점에 대해 설명해 주세요.',
'Context API를 사용하여 글로벌 상태를 관리하는 방법에 대해 설명해 주세요.',

'tech_keyword': 'tanstack query',
'TanStack Query를 사용하여 서버 상태를 관리하는 방법에 대해 설명해 주세요.',
'TanStack Query의 캐싱 메커니즘에 대해 설명해 주세요.',
'TanStack Query와 Redux를 비교하여 설명해 주세요.',
"""
followed_questions_checkpoint_preferences = """우대사항

'tech_keyword': 'ui/ux',
'사용자 경험(UX) 설계 시 가장 중요하게 고려해야 할 요소는 무엇이라고 생각하십니까?',
'UI/UX 디자인에서 사용자의 피드백을 어떻게 수집하고 반영하십니까?',
'프로젝트에서 UI/UX 디자인의 성공을 어떻게 측정하십니까?',
'사용자 인터페이스(UI)와 사용자 경험(UX)의 차이점은 무엇이라고 설명하시겠습니까?',
'UI/UX 디자인에서 접근성을 고려할 때 어떤 점을 중점적으로 보십니까?',
'UI/UX 디자인에서 최신 트렌드를 어떻게 반영하십니까?']}]}
"""
questions_wowpoint = """'Tanstack-Query를 사내에 도입하여 상태 관리의 어려움을 해결한 경험이 있다고 하셨는데, Tanstack-Query를 선택하게 된 이유와 이를 통해 해결한 구체적인 문제 사례를 설명해 주시겠어요?',
'next-auth와 AWS cognito를 통합하여 로그인 구현을 하셨다고 했습니다. 이 과정에서 TypeScript를 활용하여 얻은 이점과 직면했던 기술적 도전 과제는 무엇이었나요?',
'Storybook을 이용하여 UI 컴포넌트 단위 테스트를 진행하고 클라이언트와 소통하셨다고 했습니다. Storybook을 활용한 UI/UX 개선 사례와 클라이언트와의 소통에서 가장 큰 성과는 무엇이었나요?',
'questions_wowpoint_experience': ['프론트엔드 스쿨 플러스 단기 심화 과정에서 최종 발표 심사위원으로 참여하셨다고 했습니다. 해당 과정에서 학생들의 프로젝트를 심사하면서 겪은 문제들 중 가장 힘들었던 것과, 그 문제를 해결한 경험에 대해 소개해주세요.',
'(주) 멋쟁이사자처럼 프론트엔드 스쿨에서 특강 강사로 활동하셨다고 했습니다. 학생들에게 프로젝트 기획 방법과 협업 방법을 교육하면서 겪은 문제들 중 가장 힘들었던 것과, 그 문제를 해결한 경험에 대해 소개해주세요.
"""
questions_doubtpoint = """'스포츠마케팅학과를 졸업하신 후, 개발자로 빠르게 전환하셨습니다. 개발 관련 학습이나 경험을 쌓으신 과정이 궁금합니다. 어떤 경로로 개발 역량을 키우셨나요?',
'밸류와이즈 개발팀에서 프리랜서로 1개월간 근무하셨다고 하셨습니다. 이 프로젝트의 성격이나 목표가 무엇이었는지, 그리고 1개월이라는 짧은 기간 동안 어떤 성과를 이루셨는지 설명해 주실 수 있나요?'
"""
keywords_compared_with_keywordlist_and_jd = """"""

@router.post("/create_question/{interview_id}", response_model=CreateQuestionResponse)
async def query_search(request_model: CreateQuestionRequest, interview_id: str, question_service: InterviewService = Depends(get_interview_service)):
    # result = question_service.create_questions(request_model.portfolio_data, request_model.job_description_data, request_model.input_position)
    logger.info(
    f"Portfolio Data: {request_model.portfolio_data}, "
    f"Job Description Data: {request_model.job_description_data}, "
    f"Input Position: {request_model.input_position}"
    )

    logger.info(f"keywords_compared_with_keywordlist_and_jd : {keywords_compared_with_keywordlist_and_jd}" )
    await add_event({"type" : "keywords_compared_with_keywordlist_and_jd", "data": keywords_compared_with_keywordlist_and_jd})
    await asyncio.sleep(10)
    logger.info(f"pf_original : {pf_original}")
    await add_event({"type" : "pf_original", "data": pf_original})
    logger.info(f"jd_original : {jd_original}")
    await add_event({"type" : "jd_original", "data": jd_original})


    await asyncio.sleep(10)
    logger.info(f"conformitypoint : {conformitypoint}")
    logger.info(f"wowpoint : {wowpoint}")
    logger.info(f"doubtpoint_pf_only : {doubtpoint_pf_only}")
    await add_event({"type" : "conformitypoint", "data": conformitypoint})
    await add_event({"type" : "wowpoint", "data": wowpoint})
    await add_event({"type" : "doubtpoint_pf_only", "data": doubtpoint_pf_only})

    await asyncio.sleep(10)
    logger.info(f"weak_requirements_keywords : {weak_requirements_keywords}")
    logger.info(f"weak_preferences_keywords : {weak_preferences_keywords}")
    logger.info(f"check_requirements_keywords : {check_requirements_keywords}")
    logger.info(f"check_preferences_keywords : {check_preferences_keywords}")
    await add_event({"type" : "weak_requirements_keywords", "data": weak_requirements_keywords})
    await add_event({"type" : "weak_preferences_keywords", "data": weak_preferences_keywords})
    await add_event({"type" : "check_requirements_keywords", "data": check_requirements_keywords})
    await add_event({"type" : "check_preferences_keywords", "data": check_preferences_keywords})

    await asyncio.sleep(10)
    logger.info(f"requirements_in_pf_semantic_search : {requirements_in_pf_semantic_search}")
    logger.info(f"requirements_not_in_pf_semantic_search : {requirements_not_in_pf_semantic_search}")
    logger.info(f"preferences_in_pf_semantic_search : {preferences_in_pf_semantic_search}")
    logger.info(f"preferences_not_in_pf_semantic_search : {preferences_not_in_pf_semantic_search}")
    await add_event({"type" : "requirements_in_pf_semantic_search", "data": requirements_in_pf_semantic_search})
    await add_event({"type" : "requirements_not_in_pf_semantic_search", "data": requirements_not_in_pf_semantic_search})
    await add_event({"type" : "preferences_in_pf_semantic_search", "data": preferences_in_pf_semantic_search})
    await add_event({"type" : "preferences_not_in_pf_semantic_search", "data": preferences_not_in_pf_semantic_search})

    await asyncio.sleep(10)
    await add_event({"type" : "main_questions_weakpoint_requirements", "data": main_questions_weakpoint_requirements})
    await add_event({"type" : "main_questions_weakpoint_preferences", "data": main_questions_weakpoint_preferences})
    await add_event({"type" : "followed_questions_weakpoint_requirements", "data": followed_questions_weakpoint_requirements})
    await add_event({"type" : "followed_questions_weakpoint_preferences", "data": followed_questions_weakpoint_preferences})
    await add_event({"type" : "main_questions_checkpoint_requirements", "data": main_questions_checkpoint_requirements})
    await add_event({"type" : "main_questions_checkpoint_preferences", "data": main_questions_checkpoint_preferences})
    await add_event({"type" : "followed_questions_checkpoint_requirements", "data": followed_questions_checkpoint_requirements})
    await add_event({"type" : "followed_questions_checkpoint_preferences", "data": followed_questions_checkpoint_preferences})
    await add_event({"type" : "questions_wowpoint", "data": questions_wowpoint})
    await add_event({"type" : "questions_doubtpoint", "data": questions_doubtpoint})

    session = interview_sessions.get(interview_id)
    return CreateQuestionResponse(
            main_questions_weakpoint_requirements = {},
            main_questions_weakpoint_preferences = {},
            followed_questions_weakpoint_requirements = {},
            followed_questions_weakpoint_preferences = {},
            main_questions_checkpoint_requirements = {},
            main_questions_checkpoint_preferences = {},
            followed_questions_checkpoint_requirements = {},
            followed_questions_checkpoint_preferences = {},
            questions_wowpoint = {},
            questions_doubtpoint = {},
            keywords_compared_with_keywordlist_and_jd = {},
        )

# Technical interview questions
BE_MAINCONTENT = "User님의 면접 합격 가능성은 13%입니다. \n 면접 영상 분석 결과, 면접 준비 상태는 '부족'입니다. \n 역량 상승이 절대적으로 필요합니다. \n 답변 분석을 참고하여 본인의 면접 습관을 찾아 개선해보세요. 본인의 면접 습관을 차근차근 고친다면 면접 합격 가능성을 높일 수 있습니다."
FE_MAINCONTENT = "User님의 면접 합격 가능성은 13%입니다. \n 면접 영상 분석 결과, 면접 준비 상태는 '부족'입니다. \n 역량 상승이 절대적으로 필요합니다. \n 답변 분석을 참고하여 본인의 면접 습관을 찾아 개선해보세요. 본인의 면접 습관을 차근차근 고친다면 면접 합격 가능성을 높일 수 있습니다."

BE_QUESTIONS = [
    {"keyword": "Java", "question" : "본인의 기술 스택에 대해 설명해주세요.", "score": 1, "reason": "React를 사용하는 회사에 지원하셨으므로 React에 대한 질문을 추가했습니다."},
    {"keyword": "DB", "question" : "가장 어려웠던 기술적 문제와 해결 방법은?", "score": 1, "reason": "React를 사용하는 회사에 지원하셨으므로 React에 대한 질문을 추가했습니다."},
    {"keyword": "Architecture", "question" : "RESTful API의 특징에 대해 설명해주세요.", "score": 1, "reason": "React를 사용하는 회사에 지원하셨으므로 React에 대한 질문을 추가했습니다."},
    {"keyword": "Spring", "question" : "웹 성능 최적화 경험이 있다면 공유해주세요.", "score": 1, "reason": "React를 사용하는 회사에 지원하셨으므로 React에 대한 질문을 추가했습니다."}
]

# Behavioral interview questions
FE_QUESTIONS = [
    {"keyword": "tanstack query", "question" : "당신의 포트폴리오에 따르면, Tanstack-Query를 사내에 도입했다고 언급하셨습니다. 이 도입 과정에서 직면했던 가장 큰 도전 과제는 무엇이었으며, 이를 어떻게 해결하셨나요?.", "score": 4, "reason": "사용자는 Tanstack-Query 도입 시 기존 상태관리 라이브러리와의 호환성 문제를 해결하는 과정에서의 도전 과제를 잘 설명하였습니다. 특히, 초기 단계에서 비핵심 API 요청을 대상으로 테스트를 진행하여 문제점을 사전에 식별하고, 이를 통해 안정적인 전환을 이끌어냈다는 점에서 구체적인 예시와 세부사항을 포함하고 있습니다. 다만, 공식 문서나 팀원들과의 협업 등 추가적인 해결 방법에 대한 언급이 부족하여 5점에 미치지 못합니다.", "loss_keyword":['state management', 'tanstack query']},
    {"keyword": "tanstack query", "question" : "TanStack Query를 기존 상태 관리 라이브러리와 통합할 때 발생할 수 있는 잠재적인 문제점은 무엇이며, 이를 어떻게 해결할 수 있을까요?", "score": 0, "reason": "사용자는 질문에 대해 잘 모른다고 답변하였으며, TanStack Query와 기존 상태 관리 라이브러리 통합 시 발생할 수 있는 문제점이나 해결책에 대한 정보를 제공하지 않았습니다.", "loss_keyword":['state management', 'tanstack query']},
    {"keyword": "context api", "question" : "포트폴리오에 따르면, ContextAPI를 사용하면서 상태 관리의 어려움을 겪었다고 하셨습니다. 이 문제를 해결하기 위해 어떤 접근 방식을 사용하셨나요?", "score": 0, "reason": "사용자는 질문에 대해 잘 모르겠다고 답변하여, ContextAPI 사용 시 상태 관리의 어려움을 해결하기 위한 접근 방식에 대한 정보를 제공하지 않았습니다. 따라서 평가 기준에 따라 0점으로 평가됩니다.", "loss_keyword":['context api', 'state management']},
    {"keyword": "redux", "question" : "채용공고에서 redux를 요구하고 있는데, 포트폴리오/이력서에는 해당 내용이 존재하지 않습니다. 혹시 redux를 사용해보신 경험이나 이론적으로 알고 계신 내용이 있나요?", "score": 0, "reason": "사용자는 Redux에 대해 사용해본 경험이 없고, 이론적으로도 잘 모른다고 답변하였습니다. 이는 질문에 대한 답변으로 적절하지 않으며, Redux의 기본 개념이나 이론적인 설명조차 제공하지 못했습니다.", "loss_keyword":['redux']}
]

BE_KEYWORDS = ["Java", "DB", "Architecture", "redis", "Spring", "ci/cd"]

FE_KEYWORDS = [ "redux", "zustand", "graphql", "typescript", "context api", "tanstack query"]

questions = {
    "BE": BE_QUESTIONS,
    "FE": FE_QUESTIONS,
}

class getInterviewRequest(BaseModel):
    questions_count: int
    question: Dict
    answer: Optional[str]

class getInterviewResponse(BaseModel):
    questions:Dict

@router.post("/interviews/{interview_id}", response_model=getInterviewResponse)
async def get_interviews(interview_id: str, request: getInterviewRequest, interview_service: InterviewService = Depends(get_interview_service)):
    print(request)  # 요청 데이터 출력
    await asyncio.sleep(5)
    session = interview_sessions.get(interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if request.questions_count > 0 :
        await interview_service.evaulate_answer(request.questions_count, session.input_position)
        answer_data = {"questions_count": request.questions_count, 
               "question": request.question["question"], 
               "keyword": request.question["keyword"], 
               "score":request.question["score"], 
                "answer": request.answer}
        if session.input_position == "BE":
            answer_data["score"] = BE_QUESTIONS[request.questions_count - 1]["score"]
            answer_data["reason"] = BE_QUESTIONS[request.questions_count - 1]["reason"]
            answer_data["loss_keyword"] = BE_QUESTIONS[request.questions_count - 1].get("loss_keyword", [])
        else :
            answer_data["score"] = FE_QUESTIONS[request.questions_count - 1]["score"]
            answer_data["reason"] = FE_QUESTIONS[request.questions_count - 1]["reason"]
            answer_data["loss_keyword"] = BE_QUESTIONS[request.questions_count - 1].get("loss_keyword", [])
        await add_event({"type" : "score", "data": f"점수 : {answer_data['score']}"})
        await add_event({"type" : "reason", "data": f"이유 : {answer_data['reason']}"})
        await interview_service.get_searched_question(request.questions_count)
        session.history.append(answer_data)

    if session.input_position == "BE":
        question = BE_QUESTIONS[request.questions_count]
    else:
        question = FE_QUESTIONS[request.questions_count]
    
    return getInterviewResponse(questions=question)

class getInterviewEndRequest(BaseModel):
    questions_count: int

class getInterviewEndResponse(BaseModel):
    history:List
    keyword:List
    mainContent: str


@router.post("/interviews/{interview_id}/end", response_model=getInterviewEndResponse)
async def get_result(interview_id: str, request: getInterviewEndRequest):
    session = interview_sessions.get(interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    history = session.history[:request.questions_count]
    keyword = []
    print(history)
    if session.input_position == "BE":
        for key in BE_KEYWORDS:
            key_score = 0
            key_count = 0
            for data in history:
                if key == data["keyword"]:
                    key_score += data["score"]
                    key_count += 1
            if key_count > 0:
                key_score = key_score / key_count
            keyword.append({"keyword": key, "score": key_score})
        main_content = BE_MAINCONTENT
    else:
        for key in FE_KEYWORDS:
            key_score = 0
            for data in history:
                if key == data["keyword"]:
                    key_score += data["score"]
            keyword.append({"keyword": key, "score": key_score})
        main_content = FE_MAINCONTENT
    
    return getInterviewEndResponse(history=session.history[:request.questions_count], keyword=keyword, mainContent=main_content)