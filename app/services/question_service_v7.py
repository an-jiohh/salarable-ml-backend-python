# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Langchain
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

#config
from app.core.config import Settings, get_settings

# Pydantic
from pydantic import BaseModel, Field
from typing import List

# OPEN AI
from openai import OpenAI
from openai import AsyncOpenAI

# Vector Embedding
from kobert_transformers import get_kobert_model, get_tokenizer
import torch

import json

#loggin
import logging

#asyncio
import asyncio

#Counter
from collections import Counter

#꼬리질문
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


logger = logging.getLogger(__name__)

class QuestionServiceV7:
    def __init__(self, config:Settings) -> None:
        self.config = config
        #pinecone index 생성
        self.index = self.create_index(config)
        #kobert 모델 및 토크나이저 로드
        self.model = get_kobert_model()
        self.tokenizer = get_tokenizer()
        self.client = self.get_client(config)
        #전처리 데이터 로드
        self.unique_keywords = self.get_unique_keywords(config)
        
        self.params_konowledge_based:dict = self.get_params_konowledge_based()
        self.kwargs_knowlegde_based:dict = self.get_kwargs_knowlegde_based()
        #공통 - Weakpoint 대질문
        self.system_generate_main_questions_weakpoint = self.get_system_generate_main_questions_weakpoint()
        #공통 - Weakpoint 꼬리질문
        self.system_generate_followed_questions_weakpoint = self.get_system_generate_followed_questions_weakpoint()
        #공통 - Checkpoint 대질문
        self.system_generate_main_questions_checkpoint = self.get_system_generate_main_questions_checkpoint()
        #공통 - checkpoint 꼬리질문
        self.system_generate_followed_questions_checkpoint = self.get_system_generate_followed_questions_checkpoint()

    async def generate_response(self,system_message,user_message,params=None,kwargs=None,parser_type="list",custom_output=None):
            ### 기본 모델 파라미터 설정 ###
            default_params = {
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.9,
            }
            default_kwargs = {
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            }

            # 사용자로부터 전달받은 params와 kwargs를 반영하여 최종 파라미터 설정
            if params is None:
                params = default_params
            else:
                params = {**default_params, **params}

            if kwargs is None:
                kwargs = default_kwargs
            else:
                kwargs = {**default_kwargs, **kwargs}

            full_params = {**params, **kwargs}

            prompt = ChatPromptTemplate.from_messages([
                    ("system", "{system_input}"),
                    ("user", "{user_input}"),
            ])

            llm = ChatOpenAI(model="gpt-4o", api_key=self.config.openai_api_key, **full_params)
            messages = prompt.format_messages(
                system_input=system_message, user_input=user_message
            )

            # logger.info(messages)

            if parser_type == "json":
                response = await self.client.beta.chat.completions.parse(
                    model="gpt-4o",
                    **full_params,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=custom_output,
                )
                response = response.choices[0].message.content
            elif parser_type == "list":
                # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
                chain = prompt | llm | StrOutputParser()

                response = await chain.ainvoke(
                    {"system_input": system_message, "user_input": user_message}
                )
            return response
    
    #비동기 함수 찾아볼것
    def search_vector_db(self, _keyword, _sentence, _namespace):
        # 문장을 KoBERT 모델의 입력에 맞게 토큰화
        inputs = self.tokenizer(_sentence, return_tensors="pt", padding=True, truncation=True)

        # KoBERT로 문장 임베딩 생성 (CLS 토큰의 출력을 사용)
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용

        # 벡터 형태로 변환
        searchs = sentence_embeddings.numpy()

        # print(searchs)
        vectors_to_search = searchs[0].tolist()

        lower_keyword = _keyword.lower()
        # 필터링할 조건 설정 (예: 특정 키워드나 범위 지정)
        filter_criteria = {"tech_keyword": {"$in": [lower_keyword]}}

        result = self.index.query(
            namespace=_namespace,
            vector=vectors_to_search,
            top_k=3,
            filter=filter_criteria,
            include_values=False,
            include_metadata=True,
        )

        return result
    
    def append_semantic_search_result(self, _list, _keyword, original_sentence, _data):
        matches = _data.get("matches", [])
        for _match in matches:
            metadata = _match.get("metadata", {})
            tech_keyword = metadata.get("tech_keyword", "N/A")
            searched_sentence = metadata.get("text", "N/A")
            score = _match.get("score", "N/A")

            # _list에 original_sentence가 없는 경우 새로운 딕셔너리 구조 생성
            if original_sentence not in _list.setdefault(_keyword, {}):
                _list[_keyword][original_sentence] = []

            # 동일한 searched_sentence가 이미 있는지 확인하여 중복 방지
            if any(
                entry["searched_question"] == searched_sentence
                for entry in _list[_keyword][original_sentence]
            ):
                continue  # 중복일 경우 추가하지 않음

            # original_sentence의 하위 항목으로 searched_sentence와 score 추가
            _list[_keyword][original_sentence].append(
                {"searched_question": searched_sentence, "score": round(score, 2)}
            )

    async def preprocessing_pf(self, pf):
        SYSTEM_PREPROCESSING_PF_ORIGINAL = """
        역할(Role):
        * 당신은 주어진 txt형태의 포트폴리오를 전처리 해야합니다.

        목표(Goal):
        * 원본 내용을 최대한 유지하세요.
        * 내용을 생략하지 마세요.

        지시사항(Instructions):
        * 각종 특수기호를 제거하세요.
        * 각종 마스킹된 [link], [url], [name], ...등의 NER 태그들을 제거하세요.
        * Page 1, Page 2, ...등 Page번호를 제거하세요
        """
        preprocessed_pf = await self.generate_response(
            SYSTEM_PREPROCESSING_PF_ORIGINAL,
            pf,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "list",
        )
        return preprocessed_pf

    async def preprocessing_jd(self, jd):
        SYSTEM_PREPROCESSING_JD = """
        역할(Role):
        * 당신은 주어진 txt형태의 채용공고를, 전처리 해야합니다.

        목표(Goal):
        * 기술 면접을 위한 포트폴리오 요약임을 참고하세요.

        지시사항(Instructions):
        * 채용 공고의 자격요건과 우대조건을 위주로 정리하세요.
        * 자격요건과 우대조건은 개발자 기술 키워드를 위주로 정리하세요.

        결과 설정(Output):
        * 형식(Format):
            ## 자격요건:
            자격요건1,
            자격요건2,
            ...
            ## 우대사항:
            우대사항1,
            우대사항2,
            ...
        """
        preprocessed_jd = await self.generate_response(
            SYSTEM_PREPROCESSING_JD,
            jd,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "list",
        )
        return preprocessed_jd
    
    async def extract_keyword_from_jd(self, unique_keywords, jd):
        SYSTEM_EXTRACT_KEYWORD_FROM_JD = """
        역할(Role):
        * 당신은 주어진 채용공고에서 기술 키워드를 추출해야 합니다.

        지시사항(Instructions):
        * '사전 기술키워드'에 존재하지 않는 기술 키워드도 추가 가능합니다.
        * 채용공고의 자격요건과 우대사항의 모든 기술 키워드를 포함하세요.
        * 같은 의미이지만, 철자가 다른 경우 '사전 기술키워드'를 사용하세요.
        * 찾아낸 기술 키워드가 '사전 기술키워드'에 포함되면 'is_existed_in_keywordlist'를 True, 그렇지 않으면 False를 반환합니다.
        * requirements는 자격요건, preferences는 우대사항을 의미합니다.
        * 만약 특정 기술 키워드가 자격요건과 우대사항에 모두 들어간다면, requirements_and_preferences는 requirements로 지정하세요.
        * JSON 형식으로 반환하세요.

        입력 예시 (Input):
        * 형식(Format):
        사전 기술키워드 : {"기술 키워드"},
        채용공고 : {"채용공고"},


        결과 설정(Output):
        * 형식(Format):
            {
                "기술키워드 1": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": 'True or False,'
                },
                "기술키워드 2": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": 'True or False,'
                },

            }
        """
        USER_EXTRACT_KEYWORD_FROM_JD = f"""
        사전 기술키워드 : {unique_keywords},
        채용공고 : {jd},
        """
        ### "requirements_and_preferences" - [requirements : 자격요건 / preferences : 우대사항]
        ### "is_existed_in_keywordlist" - [True : 기술 키워드 리스트에 존재함 / False : 기술 키워드 리스트에 존재하지 않음]
        r_keywords_compared_with_keywordlist_and_jd = await self.generate_response(
            SYSTEM_EXTRACT_KEYWORD_FROM_JD,
            USER_EXTRACT_KEYWORD_FROM_JD,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "list",
        )

        cleaned_string = r_keywords_compared_with_keywordlist_and_jd.split("{", 1)[1].rsplit("}", 1)[0] #오류 포인트
        keywords_compared_with_keywordlist_and_jd = json.loads("{" + cleaned_string + "}")

        return keywords_compared_with_keywordlist_and_jd
    
    async def extract_conformitypoint(self, pf_original, keywords_compared_with_keywordlist_and_jd, tech_field):
        class SubOutputExtractConformitypoint(BaseModel):
            tech_keywords: str = Field(description="기술 키워드")
            requirements_and_preferences: str = Field(
                description="requirements or preferences"
            )
            is_existed_in_keywordlist: bool = Field(
                description="채용공고와 주어진 기술 키워드의 교집합이면 True, 채용공고에만 존재하면 False"
            )
            is_keywords_in_PF: bool = Field(
                description="사용자의 포트폴리오에 해당 기술 관련 내용이 있으면 True, 없으면 False"
            )
            sentences: List[str] = Field(
                description="사용자 포트폴리오에서의 기술 키워드 관련 문장"
            )

        class OutputExtractConformitypoint(BaseModel):
            conformitypoint: List[SubOutputExtractConformitypoint]

        SYSTEM_EXTRACT_CONFORMITYPOINT = """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 주어진 포트폴리오와 채용공고의 기술키워드를 비교해 의심스러운 포트폴리오와 채용공고의 평가항목의 부합성을 찾아내야 합니다.

        맥락(Context):

        지시사항(Instructions):
        * 기술 키워드가 나열된 문장 (ex. python, java, typescript ...)는 기술 스펙인 경향이 높으므로 절대 추가하지 마세요. 즉 단순히 표면적으로 기술 키워드가 드러난 문장을 추가하면 안됩니다.
        * 포트폴리오에 맥락이나 의미상 입력의 '채용공고의 기술 키워드'와 부합하는 부분이 존재한다면, 해당하는 모든 문장을 결과의 'sentences'에 추가하세요.
        * 기술 키워드 위주로, 부합성을 판단하고 기술키워드에 존재하는 부합성 항목과 존재하지 않는 부합성 항목을 분류하세요.
        * requirements_and_preferences, is_existed_in_keywordlist '채용공고의 기술 키워드'에서 입력된 값을 그대로 사용하세요.
        * is_keywordsInPF는 해당 기술 키워드 관련 내용이 사용자 포트폴리오에 드러나 있다면 True, 아니라면 False를 반환하세요.
        * sentences는 포트폴리오 내 해당되는 문장을 모두 반환하세요.
        * 만약 {tech_keyword}가 {"tech_field"}개발자로서 기본적으로 갖춰야 하는 내용이고 포트폴리오 내 해당 키워드가 존재하지 않는다면, is_keywords_in_PF 값은 True로, sentences는 {'tech_keyword'}를 추가해주세요.

        제약사항(Constraints):
        * 반환 형식은 오직 JSON 형식만 반환합니다.
        * 바로 파이썬 코드로 사용할 수 있게 JSON 형식에 불필요한 문자는 모두 제거하세요.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"포트폴리오"},
        채용공고의 기술 키워드 : {"tech_keyword"},
        분야 : {"tech_field"}

        """

        USER_EXTRACT_CONFORMITYPOINT = f"""
        포트폴리오 : {pf_original},
        채용공고의 기술 키워드 : {keywords_compared_with_keywordlist_and_jd},
        분야 : {tech_field}
        """
        response = await self.generate_response(
            SYSTEM_EXTRACT_CONFORMITYPOINT,
            USER_EXTRACT_CONFORMITYPOINT,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputExtractConformitypoint,
        )

        cleaned_string = response.split("{", 1)[1].rsplit("}", 1)[0]
        conformitypoint = json.loads("{" + cleaned_string + "}")

        logger.info(f"conformitypoint : {conformitypoint}")
        return conformitypoint
    
    # **Weak-Point / 대질문 / requirements**
    async def generate_main_questions_weakpoint_requirements(self, weak_requirements_keywords):
        class SubOutputGenerateMainQuestionsWeakpointRequirements(BaseModel):
            question: str = Field(description="생성된 질문")
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            question_type: str = Field(description="질문 유형 : 경험")
            purpose: str = Field(description="질문 의도")
            example: List[str] = Field(
                description="우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'"
            )
            reason: str = Field(description="질문 의도")

        class OutputGenerateMainQuestionsWeakpointRequirements(BaseModel):
            requirements: List[SubOutputGenerateMainQuestionsWeakpointRequirements]

        USER_GENERATE_MAIN_QUESTIONS_WEAKPOINT_REQUIREMENTS = f"""
        기술키워드 : {weak_requirements_keywords}
        """
        r_weakpoint_requirements = ""
        if weak_requirements_keywords:
            r_weakpoint_requirements = await self.generate_response(
                self.system_generate_main_questions_weakpoint,
                USER_GENERATE_MAIN_QUESTIONS_WEAKPOINT_REQUIREMENTS,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateMainQuestionsWeakpointRequirements,
            )
        cleaned_string = r_weakpoint_requirements.split("{", 1)[1].rsplit("}", 1)[0]
        main_questions_weakpoint_requirements = json.loads("{" + cleaned_string + "}")
        return main_questions_weakpoint_requirements
    
    # **Weak-Point / 대질문 / preferences**
    async def generate_main_questions_weakpoint_preferences(self, weak_preferences_keywords):
        class SubOutputGenerateMainQuestionsWeakpointPreferences(BaseModel):
            question: str = Field(description="생성된 질문")
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            question_type: str = Field(description="질문 유형 : 경험")
            purpose: str = Field(description="질문 의도")
            example: List[str] = Field(
                description="우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'"
            )
            reason: str = Field(description="질문 의도")

        class OutputGenerateMainQuestionsWeakpointPreferences(BaseModel):
            preferences: List[SubOutputGenerateMainQuestionsWeakpointPreferences]

        USER_GENERATE_MAIN_QUESTIONS_WEAKPOINT_PREFERENCES = f"""
        기술키워드 : {weak_preferences_keywords},
        """
        r_weakpoint_preferences = ""
        if weak_preferences_keywords:
            r_weakpoint_preferences = await self.generate_response(
                self.system_generate_main_questions_weakpoint,
                USER_GENERATE_MAIN_QUESTIONS_WEAKPOINT_PREFERENCES,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateMainQuestionsWeakpointPreferences,
            )
        cleaned_string = r_weakpoint_preferences.split("{", 1)[1].rsplit("}", 1)[0]
        main_questions_weakpoint_preferences = json.loads("{" + cleaned_string + "}")
        return main_questions_weakpoint_preferences
    
    # **Weak-Point / 꼬리질문 / requirements**
    async def generate_followed_questions_weakpoint_requirements(self, requirements_not_in_pf_semantic_search):
        class SubOutputGenerateFollowedQuestionsRequirements(BaseModel):
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            questions: List[str] = Field(description="생성된 꼬리 질문들")

        class OutputGenerateFollowedQuestionsWeakpointRequirements(BaseModel):
            requirements: List[SubOutputGenerateFollowedQuestionsRequirements]

        USER_GENERATE_FOLLOWED_QUESTIONS_WEAKPOINT_REQUIREMENTS = f"""
        Result of Similarity Search: {requirements_not_in_pf_semantic_search}
        """
        followed_questions_weakpoint_requirements = {}
        if requirements_not_in_pf_semantic_search:
            r_followed_questions_weakpoint_requirements = await self.generate_response(
                self.system_generate_followed_questions_weakpoint,
                USER_GENERATE_FOLLOWED_QUESTIONS_WEAKPOINT_REQUIREMENTS,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateFollowedQuestionsWeakpointRequirements,
            )
            cleaned_string = r_followed_questions_weakpoint_requirements.split("{", 1)[1].rsplit("}", 1)[0]
            followed_questions_weakpoint_requirements = json.loads(
                "{" + cleaned_string + "}"
            )
        else:
            followed_questions_weakpoint_requirements["requirements"] = {
                "tech_keyword": None,
                "questions": None,
            }
        return followed_questions_weakpoint_requirements
    
    # **Weak-Point / 꼬리질문 / preferences**
    async def generate_followed_questions_weakpoint_preferences(self, preferences_not_in_pf_semantic_search):
        class SubOutputGenerateFollowedQuestionsWeakpointPreferences(BaseModel):
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            questions: List[str] = Field(description="생성된 꼬리 질문들")

        class OutputGenerateFollowedQuestionsWeakpointPreferences(BaseModel):
            preferences: List[SubOutputGenerateFollowedQuestionsWeakpointPreferences]

        USER_GENERATE_FOLLOWED_QUESTIONS_WEAKPOINT_PREFERENCES = f"""
        Result of Similarity Search: {preferences_not_in_pf_semantic_search}
        """
        followed_questions_weakpoint_preferences = {}
        if preferences_not_in_pf_semantic_search:
            r_followed_questions_weakpoint_preferences = await self.generate_response(
                self.system_generate_followed_questions_weakpoint,
                USER_GENERATE_FOLLOWED_QUESTIONS_WEAKPOINT_PREFERENCES,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateFollowedQuestionsWeakpointPreferences,
            )
            cleaned_string = r_followed_questions_weakpoint_preferences.split("{", 1)[1].rsplit("}", 1)[0]
            followed_questions_weakpoint_preferences = json.loads(
                "{" + cleaned_string + "}"
            )
        else:
            followed_questions_weakpoint_preferences["preferences"] = {
                "tech_keyword": None,
                "questions": None,
            }
        return followed_questions_weakpoint_preferences
    
    # **Check-Point / 대질문 / requirements**
    async def generate_main_questions_checkpoint_requirements(self, pf_original, check_requirements_keywords):
        # 원하는 데이터 구조를 정의합니다.
        class SubOutputGenerateMainQuestionsCheckpointRequirements(BaseModel):
            question: str = Field(description="생성된 질문")
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            question_type: str = Field(description="질문 유형 : 경험")
            purpose: str = Field(description="질문 의도")
            example: List[str] = Field(
                description="우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'"
            )
            reason: str = Field(description="질문 의도")

        class OutputGenerateMainQuestionsCheckpointRequirements(BaseModel):
            requirements: List[SubOutputGenerateMainQuestionsCheckpointRequirements]

        USER_GENERATE_MAIN_QUESTIONS_CHECKPOINT_REQUIREMENTS = f"""
        포트폴리오: {pf_original},
        keywords: {check_requirements_keywords}
        """

        if check_requirements_keywords:
            r_main_questions_checkpoint_requirements = await self.generate_response(
                self.system_generate_main_questions_checkpoint,
                USER_GENERATE_MAIN_QUESTIONS_CHECKPOINT_REQUIREMENTS,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateMainQuestionsCheckpointRequirements,
            )
            cleaned_string = r_main_questions_checkpoint_requirements.split("{", 1)[1].rsplit("}", 1)[0]
            main_questions_checkpoint_requirements = json.loads("{" + cleaned_string + "}")
            return main_questions_checkpoint_requirements
        
    # **Check-Point / 대질문 / preferences**
    async def generate_main_questions_checkpoint_preferences(self, pf_original, check_preferences_keywords):
        # 원하는 데이터 구조를 정의합니다.
        class SubOutputGenerateMainQuestionsCheckpointPreferences(BaseModel):
            question: str = Field(description="생성된 질문")
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            question_type: str = Field(description="질문 유형 : 경험")
            purpose: str = Field(description="질문 의도")
            example: List[str] = Field(
                description="우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'"
            )
            reason: str = Field(description="질문 의도")

        class OutputGenerateMainQuestionsCheckpointPreferences(BaseModel):
            preferences: List[SubOutputGenerateMainQuestionsCheckpointPreferences]

        USER_GENERATE_MAIN_QUESTIONS_CHECKPOINT_PREFERENCES = f"""
        포트폴리오: {pf_original},
        keywords: {check_preferences_keywords}
        """

        if check_preferences_keywords:
            r_main_questions_checkpoint_preferences = await self.generate_response(
                self.system_generate_main_questions_checkpoint,
                USER_GENERATE_MAIN_QUESTIONS_CHECKPOINT_PREFERENCES,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateMainQuestionsCheckpointPreferences,
            )
            cleaned_string = r_main_questions_checkpoint_preferences.split("{", 1)[1].rsplit("}", 1)[0]
            main_questions_checkpoint_preferences = json.loads("{" + cleaned_string + "}")
            return main_questions_checkpoint_preferences
        
    # **Check-Point / 꼬리질문 / requirements**
    async def generate_followed_questions_checkpoint_requirements(self, requirements_in_pf_semantic_search):
        class SubOutputGenerateFollowedQuestionsCheckpointRequirements(BaseModel):
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            questions: List[str] = Field(description="생성된 꼬리 질문들")

        class OutputGenerateFollowedQuestionsCheckpointRequirements(BaseModel):
            requirements: List[SubOutputGenerateFollowedQuestionsCheckpointRequirements]

        USER_GENERATE_FOLLOWED_QUESTIONS_CHECKPOINT_REQUIREMENTS = f"""
        Result of Similarity Search: {requirements_in_pf_semantic_search}
        """
        followed_questions_checkpoint_requirements = {}
        if requirements_in_pf_semantic_search:
            r_followed_questions_checkpoint_requirements = await self.generate_response(
                self.system_generate_followed_questions_checkpoint,
                USER_GENERATE_FOLLOWED_QUESTIONS_CHECKPOINT_REQUIREMENTS,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateFollowedQuestionsCheckpointRequirements,
            )
            cleaned_string = r_followed_questions_checkpoint_requirements.split("{", 1)[1].rsplit("}", 1)[0]
            followed_questions_checkpoint_requirements = json.loads(
                "{" + cleaned_string + "}"
            )
        else:
            followed_questions_checkpoint_requirements["requirements"] = {
                "tech_keyword": None,
                "questions": None,
            }
        return followed_questions_checkpoint_requirements
    
    # **Check-Point / 꼬리질문 / preferences**
    async def generate_followed_questions_checkpoint_preferences(self, preferences_in_pf_semantic_search):
        class SubOutputGenerateFollowedQuestionsCheckpointPreferences(BaseModel):
            tech_keyword: str = Field(description="질문에 해당하는 기술 키워드")
            questions: List[str] = Field(description="생성된 꼬리 질문들")

        class OutputGenerateFollowedQuestionsCheckpointPreferences(BaseModel):
            preferences: List[SubOutputGenerateFollowedQuestionsCheckpointPreferences]

        USER_GENERATE_FOLLOWED_QUESTIONS_CHECKPOINT_PREFERENCES_JD_AND_PF = f"""
        Result of Similarity Search: {preferences_in_pf_semantic_search}
        """
        followed_questions_checkpoint_preferences = {}
        if preferences_in_pf_semantic_search:
            r_followed_questions_checkpoint_preferences = await self.generate_response(
                self.system_generate_followed_questions_checkpoint,
                USER_GENERATE_FOLLOWED_QUESTIONS_CHECKPOINT_PREFERENCES_JD_AND_PF,
                self.params_konowledge_based,
                self.kwargs_knowlegde_based,
                "json",
                OutputGenerateFollowedQuestionsCheckpointPreferences,
            )
            cleaned_string = r_followed_questions_checkpoint_preferences.split("{", 1)[1].rsplit("}", 1)[0]
            followed_questions_checkpoint_preferences = json.loads(
                "{" + cleaned_string + "}"
            )
        else:
            followed_questions_checkpoint_preferences["requirements"] = {
                "tech_keyword": None,
                "questions": None,
            }
        return followed_questions_checkpoint_preferences

    async def extract_wowpoint(self, pf_original, keywords_compared_with_keywordlist_and_jd):
        class SubExtractTechWowpoint(BaseModel):
            wowpoint: str = Field(description="기술 관련 wow-point")
            tech_keyword: List[str] = Field(description="기술 키워드")
            reason: str = Field(description="wow-point로 선정한 이유")

        class SubExtractExperienceWowpoint(BaseModel):
            wowpoint: str = Field(description="경험 관련 wow-point")
            reason: str = Field(description="wow-point로 선정한 이유")

        class OutputExtractWowpoint(BaseModel):
            wowpoint_tech: List[SubExtractTechWowpoint] = Field(
                description="기술 범주의 WOW Point"
            )
            wowpoint_experience: List[SubExtractExperienceWowpoint] = Field(
                description="경험 범주의 WOW Point"
            )

        SYSTEM_EXTRACT_WOWPOINT = """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 주어진 포트폴리오와 기술키워드를 비교해 wow-point를 찾아내야 합니다.

        맥락(Context):
        * 해당 wow-point는 포트폴리오 내 드러난 강점을 의미합니다.
        * 반환 형식은 항상 JSON 형식으로 반환하세요.

        지시사항(Instructions):
        * wow-point는 기술 범주와 경험 범주로 분류해야 합니다.
        * 기술 범주의 wow-point는 반드시 포트폴리오와 기술키워드와 관련있는 내용이되, 포트폴리오 내에 존재하는 문장이어야 합니다.
        * 기술 범주에서는 wow-point와 기술 키워드를 함께 반환하세요.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {pf_original},
        기술키워드 : {기술키워드},
        """

        USER_EXTRACT_WOWPOINT = f"""
        포트폴리오 : {pf_original},
        기술키워드 : {keywords_compared_with_keywordlist_and_jd.keys()},
        """
        r_wowpoint = await self.generate_response(
            SYSTEM_EXTRACT_WOWPOINT,
            USER_EXTRACT_WOWPOINT,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputExtractWowpoint,
        )

        cleaned_string = r_wowpoint.split("{", 1)[1].rsplit("}", 1)[0]
        wowpoint = json.loads("{" + cleaned_string + "}")
        return wowpoint
    
    async def generate_questions_wowpoint(self, wowpoint):
        # 원하는 데이터 구조를 정의합니다.
        class OutputGenerateQuestionsWowpoint(BaseModel):
            questions_wowpoint_tech: List[str] = Field(description="지식 관련 질문")
            questions_wowpoint_experience: List[str] = Field(description="경험 범주의 질문")

        SYSTEM_GENERATE_QUESTIONS_WOWPOINT = """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        지시사항(Instructions):
        * 주어진 문장마다 각각 면접 질문을 생성합니다.
        * 생성할 면접 질문은 주어진 WOW Point 기반입니다.
        1. 지식 범주의 질문은 해당 WOW Point에서 질문 가능한 가장 깊은 Depth의 질문을 제시하세요.
        2. 경험 관련 질문은 다음과 같은 Format을 사용하세요.
            Format :
            "[WOW Point]가 정말 인상적인데요. 해당 [WOW Point]를 경험하면서 겪은 문제들 중 가장 힘들었던 것과, 그 문제를 해결한 경험에 대해 소개해주세요."

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        WOW-Point:
        {
        'wowpoint_tech': [
            {'wowpoint1': '포트폴리오 내 문장1',
            'tech_keyword': ['tech_keyword A', 'tech_keyword B', ...],
            'reason': 'wowpoint 이유1'},
            {'wowpoint2': '포트폴리오 내 문장2',
            'tech_keyword': ['tech_keyword A', 'tech_keyword B', ...],
            'reason': 'wowpoint 이유2'},
            ...
            ],
        'wowpoint_experience': [
            {'wowpoint1': '포트폴리오 내 문장1',
            'reason': 'wowpoint 이유1'},
            {'wowpoint2': '포트폴리오 내 문장2',
            'reason': 'wowpoint 이유2'},
            ...
            ]
        }
        """

        USER_GENERATE_QUESTIONS_WOWPOINT = f"""
        WOW-Point: {wowpoint}
        """

        r_questions_wowpoint = await self.generate_response(
            SYSTEM_GENERATE_QUESTIONS_WOWPOINT,
            USER_GENERATE_QUESTIONS_WOWPOINT,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputGenerateQuestionsWowpoint,
        )

        cleaned_string = r_questions_wowpoint.split("{", 1)[1].rsplit("}", 1)[0]
        questions_wowpoint = json.loads("{" + cleaned_string + "}")
        return questions_wowpoint
    
    async def extract_doubtpoint_pf_only(self, pf_original):
        # 원하는 데이터 구조를 정의합니다.
        class SubOutputExtractDoubtpointPFOnly(BaseModel):
            original_sentences: str = Field(
                description="포트폴리오 내 의심스러운 정황이 포착된 문장"
            )
            reason: str = Field(description="의심스러운 이유")

        # 원하는 데이터 구조를 정의합니다.
        class OutputExtractDoubtpointPFOnly(BaseModel):
            doubtsentence_pf_only: list[SubOutputExtractDoubtpointPFOnly]

        SYSTEM_EXTRACT_DOUBTPOINT_PF_ONLY = """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 포트폴리오 자체적으로 의심스러운 정황을 모두 파악해야 합니다.

        맥락(Context):
        * 의심스러운 정황은 논리적 모순입니다.

        지시사항(Instructions):
        * 의심스러운 정황은 반드시 포트폴리오에 드러난 내용으로만 판단해야 합니다.
        * 반드시 공백 기간이 6개월 이상인 경우만, 공백기에 대한 의심스러운 정황으로 판단합니다.
        * 컴퓨터 과학 분야 전공이 아님에도, 졸업 이후 개발 관련 직군 전환이 빠르게 이루어진 경우 의심스러운 정황이 될 수 있습니다.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"포트폴리오"},

        결과 설정(Output):
        * 형식(Format):
        class SubOutputExtractDoubtpointPFOnly(BaseModel):
            original_sentences: str = Field(description="포트폴리오 내 의심스러운 정황이 포착된 문장")
            reason: str =  Field(description="의심스러운 이유")
        class OutputExtractDoubtpointPFOnly(BaseModel):
            doubtsentence_pf_only: SubOutputExtractDoubtpointPFOnly = Field(description="포트폴리오 내 문장")
        """

        USER_EXTRACT_DOUBTPOINT_PF_ONLY = f"""
        포트폴리오 : {pf_original},
        """

        r_doubtpoint_pf_only = await self.generate_response(
            SYSTEM_EXTRACT_DOUBTPOINT_PF_ONLY,
            USER_EXTRACT_DOUBTPOINT_PF_ONLY,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputExtractDoubtpointPFOnly,
        )

        cleaned_string = r_doubtpoint_pf_only.split("{", 1)[1].rsplit("}", 1)[0]
        doubtpoint_pf_only = json.loads("{" + cleaned_string + "}")
        return doubtpoint_pf_only
    
    async def generate_questions_doubtpoint(self, doubtpoint_pf_only, pf_original):
    # 원하는 데이터 구조를 정의합니다.
        class OutputGenerateQuestionsDoubtpoint(BaseModel):
            questions_doubtpoint: List[str] = Field(
                description="포트폴리오 기반 의심스러운 경향 관련 질문"
            )

        SYSTEM_GENERATE_QUESTIONS_DOUBTPOINT = """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 주어진 포트폴리오와 의심스러운 정황{doubtPoint}을 비교해 질문을 생성해야합니다.

        지시사항(Instructions):
        * 주어진 의심스러운 정황에 대해, 포트폴리오에서 의문이 해결된다면 해당 내용에 대한 질문은 생성하지 마세요.
        * 질문 생성시, 다음과 같은 Format을 참고하세요. 문장이 자연스럽지 않으면 꼭 Format을 참고하지 않아도 됩니다.
            Format :
            "A라고 이해했는데, B에서 의문점이 생깁니다. A이면 C 아닌가요?
        A는 포트폴리오에서 의심스러운 문장과 관련된 근거, B는 Doubt Point, C는 보편적인 사실을 의미합니다. (C는 자체적으로 판단하되, 구체적인 예시를 제시하세요.)

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.

        입력 예시 (Input):
        * 형식(Format):
        doubtpoint_pf_only: {doubtPoint},
        포트폴리오: {pf_original}

        결과 설정(Output):
        * 형식(Format):
        {
        "doubt_questions_pf_only": [
            질문1, 질문2, 질문3, ...
        ],
        }
        """

        USER_GENERATE_QUESTIONS_DOUBTPOINT = f"""
        doubtpoint_pf_only: {doubtpoint_pf_only},
        포트폴리오: {pf_original}
        """

        r_questions_doubtpoint = await self.generate_response(
            SYSTEM_GENERATE_QUESTIONS_DOUBTPOINT,
            USER_GENERATE_QUESTIONS_DOUBTPOINT,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputGenerateQuestionsDoubtpoint,
        )

        cleaned_string = r_questions_doubtpoint.split("{", 1)[1].rsplit("}", 1)[0]
        questions_doubtpoint = json.loads("{" + cleaned_string + "}")
        return questions_doubtpoint
    
    #꼬리질문
    #llm을 이용한 질의 확장
    async def generate_augumented_questions(self, question):
        class OutputGenerateAugumentedQuestions(BaseModel):
            questions: List[str]

        system_message = "주어진 면접 질문과 변형하여 면접 질문들을 한글로 증강해주세요. 숫자 인덱스 사용하지 말고, 각 질문은 줄바꿈으로 구분하세요"
        default_params = {
            "temperature": 0.0,
            # temperature가 1.0에 가까움 : 더 창의적이고 다양한 결과를 도출
            "max_tokens": 4096,
            # 최대 토큰 수 지정 4o모델의 경우, 4096이 최대
            "top_p": 0.1,
            # 전체 확률 분포의 상위 p%에 포함된 단어를 고려하여 출력
        }
        default_kwargs = {
            "frequency_penalty": 0.5,
            # frequency_penalty가 1.0에 가까움 : 동일한 단어의 반복적인 사용을 크게 억제
            "presence_penalty": 0.0,
            # presence_penalty가 1.0에 가까움 : 생성되는 텍스트에 새로운 주제를 도입하려고 시도함
        }
        response = await self.generate_response(
            system_message,
            question,
            default_params,
            default_kwargs,
            "json",
            OutputGenerateAugumentedQuestions,
        )
        cleaned_string = response.split("{", 1)[1].rsplit("}", 1)[0]
        parsed_response = json.loads("{" + cleaned_string + "}")
        return parsed_response["questions"]
    
    # **2. 검색 및 Top Rank 알고리즘**

    async def rank_questions(self, original_question, augumented_questions):
        vectorizer = TfidfVectorizer().fit_transform(
            [original_question] + augumented_questions
        )
        vectors = vectorizer.toarray()
        input_vector = vectors[0]
        candidate_vectors = vectors[1:]

        # 코사인 유사도 계산
        similarities = cosine_similarity([input_vector], candidate_vectors)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_questions = [
            (augumented_questions[i], similarities[i]) for i in ranked_indices
        ]
        return ranked_questions
    
    # **3. 증강된 질문에 대한 답변 생성**
    async def generate_answers(self, question):
        class OutputGenerateAnswers(BaseModel):
            question: str = Field(description="생성된 질문")
            best_answer: str = Field(description="모범답안")
            tech_keyword: str = Field(description="답변 생성시 추출한 기술키워드")
            link: str = Field(
                description="모범답안 생성시 사용한 기술키워드의 공식문서 링크"
            )
            example: List[str] = Field(
                description="우수 : '우수 응답 가이드', 보통 : '보통 응답 가이드', 미흡 : '미흡 응답 가이드'"
            )

        system_message = """
        다음 주어진 3개의 면접 질문들에 대해, 각각 1~2문장의 모범답안을 제시해주세요.
        답변 생성시 반드시 질문의 기술키워드를 추출하고, 추출된 기술키워드의 공식문서를 참고하여 답변을 생성하세요.
        또한, 주어진 질문에 대해 우수, 보통, 미흡 응답 가이드를 작성하세요.
        응답 가이드는 사용자의 답변을 평가하기 위한 용도로 사용될 예정입니다.
        * json 형식으로 출력하세요.
        """
        response = await self.generate_response(
            system_message,
            question,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputGenerateAnswers,
        )
        cleaned_string = response.split("{", 1)[1].rsplit("}", 1)[0]
        answer = json.loads("{" + cleaned_string + "}")

        # response = generate_response(system_message, questions, "list")
        # print(f"그냥 응답 {response}")
        # # 불필요한 부분 제거 및 JSON 파싱
        # cleaned_string = response.split("{", 1)[1].rsplit("}", 1)[0]
        # QnA = json.loads("{" + cleaned_string + "}")
        # print(f"json형식 변환 :{QnA}")
        return answer

    # **4. 평가점수 산출**
    async def evaluate_answer(self, original_question, original_answer, QnA, unique_keywords):
        # 원하는 데이터 구조를 정의합니다.
        class OutputEvaluateQuestion(BaseModel):
            score: int = Field(description="0~5점 사이의 평가 점수")
            reason: str = Field(description="평가 이유")
            tech_keywords: List[str] = Field(description="부족한 키워드 리스트")

        system_message = """
        당신은 FE, BE 면접을 담당하는 면접관입니다.
        주어진 면집 질문에 대해서, 0~5점 사이의 평가 점수와 평가 이유를 제시하세요.

        * 부족한 기술 키워드는 입력된 '사전 기술 키워드'에서 찾아서 사용하세요.
        * '사용자 답변'을 '참고 데이터'와 비교하여 '사용자 답변'을 평가하세요.
        * 평가 척도는 입력의 '응답 가이드'를 참고하세요.
        * 평가 척도는 다음과 같습니다.
            0점 : 대답이 아예 다른 내용이거나 모른다고 대답함.
            1점 : 대답이 불충분하거나 주제에서 벗어남.
            2점 : 미흡에 해당하는 답변 제공.
            3점 : 보통에 해당하는 답변으로 대부분의 기본 내용을 포함하나, 구체적인 예시나 세부 사항이 부족함.
            4점 : 구체적인 예시나 세부사항을 포함했으나, 그 답변의 질이 부족함.
            5점 : 전문적 지식 및 구체적인 경험을 바탕으로 깊이 있는 답변 제공.
        * json 형식으로 출력하세요.
        """
        user_message = f"질문 : {original_question}, 사용자 답변 : {original_answer}, 참고 데이터 : {QnA}, 사전 기술 키워드 : {unique_keywords}"
        # r_score = generate_response(system_message, user_message)

        # cleaned_string = r_score.split("{", 1)[1].rsplit("}", 1)[0]
        # score = json.loads("{" + cleaned_string + "}")

        response = await self.generate_response(
            system_message,
            user_message,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputEvaluateQuestion,
        )
        cleaned_string = response.split("{", 1)[1].rsplit("}", 1)[0]
        evaluation = json.loads("{" + cleaned_string + "}")
        return evaluation
    
    async def evaluation_question(self, input_question, input_answer):
        ### 1. LLM을 이용한 질의 확장 ###
        expanded_questions = await self.generate_augumented_questions(input_question)

        print(f"expanded_questions:{expanded_questions}")

        ### 2. 검색 및 Top Rank 알고리즘 ###
        ranked_questions = await self.rank_questions(input_question, expanded_questions)

        print(f"ranked_questions:{ranked_questions}")

        ### 3. 증강된 질문에 대한 답변 생성 ###
        augumented_datas = []
        for generated_question in [q[0] for q in ranked_questions[:3]]:
            augumented_datas.append(
                await self.generate_answers(generated_question)
            )  # 상위 3개 질문에 대해 답변 생성

        print(f"augumented_datas: {augumented_datas}")
        print("-" * 200)

        ### 4. 답변 리스트 작성 및 Top Rank 재정렬 ###
        # ranked_answers = rank_answers(input_question, generated_answers)

        # prompt_answer = []
        # for answer in ranked_answers:
        #   prompt_answer.append(answer[1])

        ### 4. 평가점수 산출 ###
        evaluation = await self.evaluate_answer(
            input_question, input_answer, augumented_datas, self.unique_keywords
        )

        # 결과 출력
        print(f"점수 : {evaluation['score']}")
        print(f"평가 이유 : {evaluation['reason']}")
        print(f"부족한 기술 키워드 : {evaluation['tech_keywords']}")
        print("-" * 200)

        return evaluation
    
    async def followed_question(self, q, history, namespace):
        class OutputFollowedQuestion(BaseModel):
            question: str

        SYSTEM_FOLLOWED_QUESTION = """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 기술 꼬리 질문을 하나 생성해야합니다.
        * 벡터 DB내 Similarity Search를 통한 결과물인 실제 면접 질문을 참고하여 적절한 기술 면접 질문을 생성합니다.

        지시사항(Instructions):
        * 입력된 'history'와 'Result of Similarity Search'를 참고하여, 꼬리질문을 하나 생성하세요.
        * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]를 그대로 사용합니다.
        * 출력에서 key는 항상 tech_keyword로 사용하세요.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        history: {history},
        Result of Similarity Search:
        {
        '기술키워드1': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ],
        '기술키워드2': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ]
        }
        """

        searched_datasets = []
        for tech_keyword in q["tech_keywords"]:
            searched_data = self.search_vector_db(tech_keyword, tech_keyword, namespace)
            for m in searched_data["matches"]:
                searched_datasets.append(m["metadata"]["text"])

        USER_FOLLOWED_QUESTION = f"""
        history: {history},
        Result of Similarity Search : {searched_datasets}
        """
        r_followed_questions = await self.generate_response(
            SYSTEM_FOLLOWED_QUESTION,
            USER_FOLLOWED_QUESTION,
            self.params_konowledge_based,
            self.kwargs_knowlegde_based,
            "json",
            OutputFollowedQuestion,
        )
        # 불필요한 부분 제거 및 JSON 파싱
        cleaned_string = r_followed_questions.split("{", 1)[1].rsplit("}", 1)[0]
        json_data = json.loads("{" + cleaned_string + "}")
        return searched_datasets, json_data["question"]


    @staticmethod
    def create_index(config:Settings):
        pc = Pinecone(api_key = config.pinecone_api_key)
        index = pc.Index(config.pinecone_index_name)
        return index
    
    @staticmethod
    def get_client(config:Settings):
        return AsyncOpenAI(api_key=config.openai_api_key)
    
    @staticmethod
    def get_unique_keywords(config:Settings):
        with open(config.pre_processed_dataset, 'r',  encoding='utf-8') as file:
            data = json.load(file)
        # 특정 컬럼 데이터 조회
        # json_uuid_list = data["uuid_list"]
        # json_questions = data["questions"]
        # json_embeddings = data["embeddings"]
        # json_tech_fields = data["tech_fields"]
        json_tech_keywords = data["tech_keywords"]

        # df_loaded = pd.DataFrame(data)

        # 개수 카운팅 및 중복 없이 리스트 반환
        # 플랫 리스트 생성 및 카운팅
        flat_list = [item for sublist in json_tech_keywords for item in sublist]
        keyword_counts = Counter(flat_list)
        unique_keywords = list(keyword_counts.keys())
        unique_keywords.sort()

        return unique_keywords
    
    @staticmethod
    def get_params_konowledge_based():
        return {
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1.0,
        }
    
    @staticmethod
    def get_kwargs_knowlegde_based():
        return {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    
    @staticmethod
    def get_system_generate_main_questions_weakpoint():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 대질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 생성하세요.
        1. 질문 의도는 이 사용자가 해당 기술을 사용해 봤는지에 대한 사실 확인 목적입니다.
        2. 질문 유형은 경험적인 질문으로 생성하세요.
        3. 다음과 같은 Format을 참고하세요. "채용공고에서 {tech_keyword}를 요구하고 있는데, 포트폴리오/이력서에는 해당 내용이 존재하지 않습니다. 혹시 {tech_keyword}를 사용해보신 경험이나 이론적으로 알고 계신 내용이 있나요?"
        * 출력의 {keyword}는 반드시 입력의 '기술키워드 : {"tech_keywords"}'만 사용하세요.
        * 출력의 example은 면접에 대한 사용자의 응답 가이드입니다. 이는 사용자 답변의 평가 기준으로 사용할 예정입니다.
        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        기술키워드 : {"tech_keywords"}
        """
    
    @staticmethod
    def get_system_generate_followed_questions_weakpoint():
        return """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        * 벡터 DB내 Similarity Search를 통한 결과물인 실제 면접 질문을 참고하여 적절한 면접 질문을 생성합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 3개 이상 생성하세요.
        * 검색된 질문이 해당 기술키워드와 관련이 없다면 변형해서 관련있는 질문을 생성하세요.
        * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]를 그대로 사용합니다.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        Result of Similarity Search:
        {
        '기술키워드1': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ],
        '기술키워드2': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ]
        }

        """
    
    @staticmethod
    def get_system_generate_main_questions_checkpoint():
        return """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 대질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 생성하세요.
        1. 질문 의도는 이 사용자가 해당 기술을 사용해 봤는지에 대한 사실 확인 목적입니다.
        2. 질문 유형은 경험적인 질문으로 생성하세요.

        * 출력의 {tech_keyword}는 반드시 입력의 'keywords : {"기술키워드"}'만 사용하세요.
        * 출력의 example은 면접에 대한 사용자의 응답 가이드입니다. 이는 사용자 답변의 평가 기준으로 사용할 예정입니다.
        * 입력된 'pf_original'를 통해 생성된 질문이 답변 가능하지 않은 질문만 생성하세요.
        * 가능하다면, 프로젝트명을 언급하세요.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        포트폴리오: {pf_original},
        keywords:{
                    '기술키워드1': {'requirements_and_preferences': 'requirements',
                                'is_existed_in_keywordlist': True/False,
                                'is_keywords_in_PF': True,
                                'sentences': ['포트폴리오 내 문장1',
                                                '포트폴리오 내 문장2',
                                                ...]},
                    '기술키워드2': {'requirements_and_preferences': 'requirements',
                                'is_existed_in_keywordlist': True/False,
                                'is_keywords_in_PF': True,
                                'sentences': ['포트폴리오 내 문장1',
                                                '포트폴리오 내 문장2',
                                                ...]},
                    ...
                    }

        결과 설정(Output):
        * 형식(Format) :
            "1": {
            question : '생성된 질문',
            tech_keyword : '질문에 해당하는 기술 키워드들',
            question_type : '질문 종류 : 경험',
            purpose: str : '질문 의도',
            example: [우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'],
            reason : '질문 생성 이유'
            },
            "2": {
            question : '생성된 질문',
            tech_keyword : '질문에 해당하는 기술 키워드들',
            question_type : '질문 종류 : 경험',
            purpose: str : '질문 의도',
            example: [우수 : '우수 답변 예시', 보통 : '보통 답변 예시', 미흡 : '미흡 답변 예시'],
            reason : '질문 생성 이유'
            },
            ...
        """
    
    @staticmethod
    def get_system_generate_followed_questions_checkpoint():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        * 벡터 DB내 Similarity Search를 통한 결과물인 실제 면접 질문을 참고하여 적절한 면접 질문을 생성합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 3개 이상 생성하세요.
        * 검색된 질문이 해당 기술키워드와 관련이 없다면 변형해서 관련있는 질문을 생성하세요.
        * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]를 그대로 사용합니다.
        * 출력에서 key는 항상 tech_keyword로 사용하세요.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.


        입력 예시 (Input):
        * 형식(Format):
        Result of Similarity Search:
        {
        '기술키워드1': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ],
        '기술키워드2': [
            {'original_sentence': '원본 질문 1',
            'searched_question': '원본 질문 1에 대하여 검색된 질문 1',
            'score': '유사도 점수'},
            {'original_sentence': '원본 질문 2',
            'searched_question': '원본 질문 2에 대하여 검색된 질문 2',
            'score': '유사도 점수'},
            ...
            ]
        }
        """
    
question_service_v7 = QuestionServiceV7(config=get_settings())

def get_question_service_v7() :
    yield question_service_v7