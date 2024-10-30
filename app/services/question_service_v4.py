from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from app.core.config import Settings, get_settings

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# Vector Embedding
from kobert_transformers import get_kobert_model, get_tokenizer
import torch

import json
from collections import Counter

#loggin
import logging

logger = logging.getLogger(__name__)

class QuestionsResponse(BaseModel):
    questions_wowpoint:dict
    questions_doubtpoint:dict
    questions_requirements_in_pf_semantic_search:dict
    questions_preferences_in_pf_semantic_search:dict
    questions_requirements_not_in_pf_semantic_search:dict
    questions_preferences_not_in_pf_semantic_search:dict

class QuestionServiceV4:
    def __init__(self, config:Settings):
        self.index = self.create_index(config)
        self.config = config
        self.model = get_kobert_model()
        self.tokenizer = get_tokenizer()
        pre_processed_dataset = self.get_pre_processed_dataset(config)
        # self.json_questions = pre_processed_dataset['questions']
        # self.json_embeddings = pre_processed_dataset['embeddings']
        # self.json_uuid_list = pre_processed_dataset['uuid_list']
        # self.json_tech_fields = pre_processed_dataset['tech_fields']
        self.json_tech_keywords = pre_processed_dataset['tech_keywords']
        self.unique_keywords = self.get_unique_keywords(self.json_tech_keywords)
        self.params_konowledge_based:dict = self.get_params_konowledge_based()
        self.kwargs_knowlegde_based:dict = self.get_kwargs_knowlegde_based()
        self.system_preprocessing_pf_original:str = self.get_system_preprocessing_pf_original()
        self.system_preprocessing_jd:str = self.get_system_preprocessing_jd()
        self.system_extract_keyword_from_jd:str = self.get_system_extract_keyword_from_jd()
        self.system_extract_wowpoint:str= self.get_system_extract_wowpoint()
        self.system_generate_questions_wowpoint:str = self.get_system_generate_questions_wowpoint()
        self.system_extract_doubtpoint_pf_only:str = self.get_system_extract_doubtpoint_pf_only()
        self.system_extract_doubtpoint_jd_and_pf:str = self.get_system_extract_doubtpoint_jd_and_pf()
        self.system_generate_questions_doubtpoint:str = self.get_system_generate_questions_doubtpoint()
        self.system_extract_conformitypoint:str = self.get_system_extract_conformitypoint()
        self.system_generate_questions_conformitypoint_jd_and_pf:str = self.get_system_generate_questions_conformitypoint_jd_and_pf()
        self.system_generate_questions_comformitypoint_jd_only:str = self.get_system_generate_questions_comformitypoint_jd_only()

    # def create_questions(self, id: str, links: list[str]) -> list[str]:
    def create_questions(self, portfolio_data: str, job_description_data: str, input_position: str) -> list[str]:

        pf_original = self.generate_response(self.system_preprocessing_pf_original, portfolio_data, "list", self.params_konowledge_based, self.kwargs_knowlegde_based)
        logger.info(pf_original)
        jd = self.generate_response(self.system_preprocessing_jd, job_description_data, "list",self.params_konowledge_based, self.kwargs_knowlegde_based)
        logger.info(f"jd:{jd}")

        class SubExtractKeywordFromJD(BaseModel):
            requirements_and_preferences: str = Field(description="requirements or preferences")
            is_existed_in_keywordlist: bool = Field(description="채용공고와 주어진 기술 키워드의 교집합이면 True, 채용공고에만 존재하면 False")

        class OutputExtractKeywordFromJD(BaseModel):
            tech_keywords: SubExtractKeywordFromJD = Field(description="채용공고에서 찾아낸 기술 키워드들")
        
        USER_EXTRACT_KEYWORD_FROM_JD = f"""
            사전 기술키워드 : {self.unique_keywords},
            채용공고 : {jd},
            """
        
        keywords_compared_with_keywordlist_and_jd = self.generate_response(self.system_extract_keyword_from_jd, USER_EXTRACT_KEYWORD_FROM_JD, "json", self.params_konowledge_based, self.kwargs_knowlegde_based,OutputExtractKeywordFromJD)

        logger.info(f"keywords_compared_with_keywordlist_and_jd:{keywords_compared_with_keywordlist_and_jd}")
        #compare PF with JD
        #wow point
        class SubExtractTechWowpoint(BaseModel):
            wowpoint: str = Field(description="기술 관련 wow-point")
            tech_keyword: List[str] = Field(description="기술 키워드")
            reason: str = Field(description="wow-point로 선정한 이유")

        class SubExtractExperienceWowpoint(BaseModel):
            wowpoint: str = Field(description="경험 관련 wow-point")
            reason: str = Field(description="wow-point로 선정한 이유")

        class OutputExtractWowpoint(BaseModel):
            wowpoint_tech: List[SubExtractTechWowpoint] = Field(description="기술 범주의 WOW Point")
            wowpoint_experience: List[SubExtractExperienceWowpoint] = Field(description="경험 범주의 WOW Point")
        
        USER_EXTRACT_WOWPOINT = f"""
        포트폴리오 : {pf_original},
        기술키워드 : {keywords_compared_with_keywordlist_and_jd.keys()},
        """

        wowpoint = self.generate_response(self.system_extract_wowpoint, USER_EXTRACT_WOWPOINT,  "json", self.params_konowledge_based, self.kwargs_knowlegde_based,OutputExtractWowpoint)

        class OutputGenerateQuestionsWowpoint(BaseModel):
            questions_wowpoint_tech: List[str] = Field(description="지식 관련 질문")
            questions_wowpoint_experience: List[str] = Field(description="경험 범주의 질문")
        USER_GENERATE_QUESTIONS_WOWPOINT = f"""
            WOW-Point: {wowpoint}
            """
        questions_wowpoint = self.generate_response(self.system_generate_questions_wowpoint, USER_GENERATE_QUESTIONS_WOWPOINT, "json",self.params_konowledge_based, self.kwargs_knowlegde_based,  OutputGenerateQuestionsWowpoint)

        #의심스러운 정황
        class SubOutputExtractDoubtpointPFOnly(BaseModel):
            original_sentences: str = Field(description="포트폴리오 내 의심스러운 정황이 포착된 문장")
            reason: str =  Field(description="의심스러운 이유")
        # 원하는 데이터 구조를 정의합니다.
        class OutputExtractDoubtpointPFOnly(BaseModel):
            doubtsentence_pf_only: SubOutputExtractDoubtpointPFOnly = Field(description="포트폴리오 내 문장")
        
        USER_EXTRACT_DOUBTPOINT_PF_ONLY = f"""
            포트폴리오 : {pf_original},
            """
        doubtpoint_pf_only = self.generate_response(self.system_extract_doubtpoint_pf_only, USER_EXTRACT_DOUBTPOINT_PF_ONLY,"json", self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputExtractDoubtpointPFOnly)

        class OutputExtractDoubtpointJDandPF(BaseModel):
            doubtsentence_jd_and_pf: List[str] = Field(description="포트폴리오 내 문장")
        
        USER_EXTRACT_DOUBTPOINT_JD_AND_PF = f"""
            포트폴리오 : {pf_original},
            채용공고 : {jd}
            """
        
        doubtpoint_jd_and_pf = self.generate_response(self.system_extract_doubtpoint_jd_and_pf, USER_EXTRACT_DOUBTPOINT_JD_AND_PF, "json",self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputExtractDoubtpointJDandPF)

        class OutputGenerateQuestionsDoubtpoint(BaseModel):
            doubt_questions_pf_only: List[str] = Field(description="포트폴리오 기반 의심스러운 경향 관련 질문")
            doubt_questions_jd_and_pf: List[str] = Field(description="채용공고 기반 의심스러운 경향 관련 질문")

        USER_GENERATE_QUESTIONS_DOUBTPOINT = f"""
        doubtpoint_pf_only: {doubtpoint_pf_only},
        doubtpoint_jd_and_pf: {doubtpoint_jd_and_pf},
        포트폴리오: {pf_original}
        """

        questions_doubtpoint = self.generate_response(self.system_generate_questions_doubtpoint, USER_GENERATE_QUESTIONS_DOUBTPOINT,  "json",self.params_konowledge_based,self.kwargs_knowlegde_based, OutputGenerateQuestionsDoubtpoint)

        #(PF && 채용공고)의 부합성
        class SubOutputExtractConformitypoint(BaseModel):
            requirements_and_preferences: str = Field(description="requirements or preferences")
            is_existed_in_keywordlist: bool = Field(description="채용공고와 주어진 기술 키워드의 교집합이면 True, 채용공고에만 존재하면 False")
            is_keywords_in_PF: bool = Field(description="사용자의 포트폴리오에 해당 기술 관련 내용이 있으면 True, 없으면 False")
            sentences: list = Field(description="사용자 포트폴리오에서의 기술 키워드 관련 문장")

        class OutputExtractConformitypoint(BaseModel):
            tech_keywords: SubOutputExtractConformitypoint = Field(description="채용공고에서 찾아낸 기술 키워드들")
        
        USER_EXTRACT_CONFORMITYPOINT = f"""
            포트폴리오 : {pf_original},
            채용공고의 기술 키워드 : {keywords_compared_with_keywordlist_and_jd},
            """
        conformitypoint = self.generate_response(self.system_extract_conformitypoint, USER_EXTRACT_CONFORMITYPOINT, "json",self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputExtractConformitypoint)
        logger.info(f"conformitypoint{conformitypoint}")
        requirements_in_pf_semantic_search = {}
        requirements_not_in_pf_semantic_search = {}

        # 우대사항
        preferences_in_pf_semantic_search = {}
        preferences_not_in_pf_semantic_search = {}

        for keyword in conformitypoint.keys():
            _type = conformitypoint[keyword]['requirements_and_preferences']
            is_in_pf = conformitypoint[keyword]['is_keywords_in_PF']
            is_in_keyword_list = conformitypoint[keyword]['is_existed_in_keywordlist']
            original_sentence = conformitypoint[keyword]['sentences']

            # 자격요건
            if _type == 'requirements':
                # pf에 존재 O
                if is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        for s in original_sentence:
                            searched_data = self.search_vector_db(keyword, s, input_position)
                            self.append_semantic_search_result(requirements_in_pf_semantic_search, keyword, s, searched_data)
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass
                # pf에 존재 X
                elif not is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        searched_data = self.search_vector_db(keyword, keyword, input_position)
                        self.append_semantic_search_result(requirements_not_in_pf_semantic_search, keyword, original_sentence, searched_data)
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass

        # 우대사항
            elif _type == 'preferences':
                # pf에 존재 O
                if is_in_pf:
                    # Semantic Search
                    if is_in_keyword_list:
                        for s in original_sentence:
                            searched_data = self.search_vector_db(keyword, s, input_position)
                            self.append_semantic_search_result(preferences_in_pf_semantic_search, keyword, s, searched_data)
                    # Wiki Search
                    elif not is_in_keyword_list:
                        pass
                # pf에 존재 X
                # Semantic Search
                if is_in_keyword_list:
                    searched_data = self.search_vector_db(keyword, keyword, input_position)
                    self.append_semantic_search_result(preferences_not_in_pf_semantic_search, keyword, original_sentence, searched_data)
                # Wiki Search
                elif not is_in_keyword_list:
                    pass

        logger.info(f"requirements_in_pf_semantic_search{requirements_in_pf_semantic_search}")
        logger.info(f"requirements_not_in_pf_semantic_search{requirements_not_in_pf_semantic_search}")
        logger.info(f"preferences_in_pf_semantic_search{preferences_in_pf_semantic_search}")
        logger.info(f"preferences_not_in_pf_semantic_search{preferences_not_in_pf_semantic_search}")
        
        #Generate Questions
        #채용공고O + 포트폴리오O : Semantic Search / 이론 & 경험 질문

        class SubOutputGenerateQuestionsConformitypointJDandPF(BaseModel):
            experience_based_questions: list = Field(description="경험 관련 질문")
            tech_based_questions: list = Field(description="이론/기술 관련 질문")
        class OutputGenerateQuestionsConformitypointJDandPF(BaseModel):
            tech_keyword: SubOutputGenerateQuestionsConformitypointJDandPF = Field(description="생성된 질문들")
        
        USER_GENERATE_QUESTIONS_CONFORMITYPOINT_REQUIREMENTS_JD_AND_PF = f"""
        포트폴리오: {pf_original},
        Result of Similarity Search: {requirements_in_pf_semantic_search}
        """

        # 자격요건 + 채용공고 O + 포트폴리오 O
        questions_requirements_in_pf_semantic_search = {}
        if requirements_in_pf_semantic_search : questions_requirements_in_pf_semantic_search = self.generate_response(self.system_generate_questions_conformitypoint_jd_and_pf, USER_GENERATE_QUESTIONS_CONFORMITYPOINT_REQUIREMENTS_JD_AND_PF, "json", self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputGenerateQuestionsConformitypointJDandPF)

        # 우대사항 + 채용공고 O + 포트폴리오 O
        USER_GENERATE_QUESTIONS_CONFORMITYPOINT_PREFERENCES_JD_AND_PF = f"""
        포트폴리오: {pf_original},
        Result of Similarity Search: {preferences_in_pf_semantic_search}
        """
        questions_preferences_in_pf_semantic_search = {}
        if preferences_in_pf_semantic_search : questions_preferences_in_pf_semantic_search = self.generate_response(self.system_generate_questions_conformitypoint_jd_and_pf, USER_GENERATE_QUESTIONS_CONFORMITYPOINT_PREFERENCES_JD_AND_PF,  "json", self.params_konowledge_based,self.kwargs_knowlegde_based, OutputGenerateQuestionsConformitypointJDandPF)

        # 채용공고O + 포트폴리오X : Semantic Search / 경험 질문
        class OutputGenerateQuestionsConformitypointJDOnly(BaseModel):
            tech_keyword: list = Field(description="생성된 질문들")
        USER_GENERATE_QUESTIONS_CONFORMITYPOINT_REQUIREMENTS_JD_ONLY = f"""
        Result of Similarity Search: {requirements_not_in_pf_semantic_search}
        """
        questions_requirements_not_in_pf_semantic_search = {}
        if requirements_not_in_pf_semantic_search : questions_requirements_not_in_pf_semantic_search = self.generate_response(self.system_generate_questions_comformitypoint_jd_only, USER_GENERATE_QUESTIONS_CONFORMITYPOINT_REQUIREMENTS_JD_ONLY,"json", self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputGenerateQuestionsConformitypointJDOnly)
        # 우대사항 + 채용공고 O + 포트폴리오 X
        USER_GENERATE_QUESTIONS_CONFORMITYPOINT_PREFERENCES_JD_ONLY = f"""
        Result of Similarity Search: {preferences_not_in_pf_semantic_search}
        """
        questions_preferences_not_in_pf_semantic_search = {}
        if preferences_not_in_pf_semantic_search : questions_preferences_not_in_pf_semantic_search = self.generate_response(self.system_generate_questions_comformitypoint_jd_only, USER_GENERATE_QUESTIONS_CONFORMITYPOINT_PREFERENCES_JD_ONLY, "json",self.params_konowledge_based,self.kwargs_knowlegde_based,  OutputGenerateQuestionsConformitypointJDOnly)

            
        logger.info(f"response wowpoint : {questions_wowpoint}")
        logger.info(f"response questions_doubtpoint : {questions_doubtpoint}")
        logger.info(f"questions_requirements_in_pf_semantic_search : {questions_requirements_in_pf_semantic_search}")
        logger.info(f"questions_preferences_in_pf_semantic_search : {questions_preferences_in_pf_semantic_search}")
        logger.info(f"questions_requirements_not_in_pf_semantic_search : {questions_requirements_not_in_pf_semantic_search}")
        logger.info(f"questions_preferences_not_in_pf_semantic_search : {questions_preferences_not_in_pf_semantic_search}")

        response = QuestionsResponse(
            questions_wowpoint=questions_wowpoint,
            questions_doubtpoint=questions_doubtpoint,
            questions_requirements_in_pf_semantic_search=questions_requirements_in_pf_semantic_search,
            questions_preferences_in_pf_semantic_search=questions_preferences_in_pf_semantic_search,
            questions_requirements_not_in_pf_semantic_search=questions_requirements_not_in_pf_semantic_search,
            questions_preferences_not_in_pf_semantic_search=questions_preferences_not_in_pf_semantic_search
        )
        return response
    
    def search_vector_db(self, keyword, sentence, namespace):

        # 문장을 KoBERT 모델의 입력에 맞게 토큰화
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        # KoBERT로 문장 임베딩 생성 (CLS 토큰의 출력을 사용)
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용

        # 벡터 형태로 변환
        searchs = sentence_embeddings.numpy()

        # logger.info(searchs)
        vectors_to_search = searchs[0].tolist()

        # 필터링할 조건 설정 (예: 특정 키워드나 범위 지정)
        filter_criteria = {
            "tech_keyword": {"$in": [keyword]}
        }

        query_results = self.index.query(
            namespace=namespace,
            vector=vectors_to_search,
            top_k=5,
            filter=filter_criteria,
            include_values=False,
            include_metadata=True
        )

        return query_results
    
    def generate_response(self, system_message, user_message, parser_type, params=None, kwargs=None, custom_output=None):

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

            ### 프롬프트 생성 ###
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_input}"),
                ("user", "{user_input}"),
            ])

            llm = ChatOpenAI(model="gpt-4o", api_key=self.config.openai_api_key, **full_params)
            messages = prompt.format_messages(system_input=system_message, user_input=user_message)

            logger.info(messages)

            if parser_type == "json":
                # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
                parser = JsonOutputParser(pydantic_object=custom_output)
                prompt = prompt.partial(format_instructions=parser.get_format_instructions())
                ### LLM Chain 생성 및 질문 생성 ###
                chain = prompt | llm | parser

            elif parser_type == "list":
                # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
                chain = prompt | llm | StrOutputParser()

            generated_response = chain.invoke({"system_input": system_message, "user_input": user_message})

            return generated_response
    
    def append_semantic_search_result(self, _list, _keyword, original_sentence, _data):
        matches = _data.get('matches', [])
        for _match in matches:
            metadata = _match.get('metadata', {})
            tech_keyword = metadata.get('tech_keyword', 'N/A')
            searched_sentences = metadata.get('text', 'N/A')
            score = _match.get('score', 'N/A')
        _list.setdefault(_keyword, []).append({
            "original_sentence" : original_sentence,
            "searched_question": searched_sentences,
            "score": round(score, 2)
        })
    
    def generate_userprompt_gen_question(self, pf, result_of_similarity_search):
            user_message = f"""
            사용자 포트폴리오: {pf},
            Result of Similarity Search: {result_of_similarity_search}
            """
            return user_message
    
    def generate_userprompt_extract_sentences_from_pf(self, unique_keywords, jd, pf):
        user_message = f"""
            기술 키워드 리스트 : {unique_keywords},
            채용 공고 : {jd},
            포트폴리오 : {pf}
            """
        return user_message
    
    @staticmethod
    def create_index(config:Settings):
        pc = Pinecone(api_key = config.pinecone_api_key)
        index = pc.Index(config.pinecone_index_name)
        return index
    
    @staticmethod
    def get_pre_processed_dataset(config:Settings):
        with open(config.pre_processed_dataset, 'r',  encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def get_unique_keywords(json_tech_keywords):
        flat_list = [item for sublist in json_tech_keywords for item in sublist]
        keyword_counts = Counter(flat_list)
        unique_keywords = list(keyword_counts.keys())
        return unique_keywords
    

    @staticmethod
    def get_params_konowledge_based():
        return {
            "temperature": 0.0, # 0.0 -> 0.7
            "max_tokens": 4096,
            "top_p": 1.0, # 1.0 -> 0.5
        }
    
    @staticmethod
    def get_kwargs_knowlegde_based():
        return {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    
    @staticmethod
    def get_system_preprocessing_pf_original():
        return """
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
    @staticmethod
    def get_system_preprocessing_jd():
        return """
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
    @staticmethod
    def get_system_extract_keyword_from_jd():
        return """
        역할(Role):
        * 당신은 주어진 채용공고에서 기술 키워드를 추출해야 합니다.

        지시사항(Instructions):
        * '사전 기술키워드'에 존재하지 않는 기술 키워드도 추가 가능합니다.
        * 채용공고의 자격요건과 우대사항의 모든 기술 키워드를 포함하세요.
        * 같은 의미이지만, 철자가 다른 경우 '사전 기술키워드'를 사용하세요.
        * 찾아낸 기술 키워드가 '사전 기술키워드'에 포함되면 'is_existed_in_keywordlist'를 True, 그렇지 않으면 False를 반환합니다.
        * requirements는 자격요건, preferences는 우대사항을 의미합니다.
        * JSON 형식으로 반환하세요.
        * You must respond only with JSON data without wrapping it in code blocks or any other text.

        입력 예시 (Input):
        * 형식(Format):
        사전 기술키워드 : {"기술 키워드"},
        채용공고 : {"채용공고"},


        결과 설정(Output):
        * 형식(Format):
            {
                "기술키워드 1": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": true or false,'
                },
                "기술키워드 2": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": 'true or false,'
                },

            }
        """
    @staticmethod
    def get_system_generate_questions_wowpoint():
        return """
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
        * 반환 형식은 오직 JSON 형식만 반환합니다.


        입력 예시 (Input):
        * 형식(Format):
        WOW-Point: {wowpoint}

        결과 설정(Output):
        * 형식(Format):
        {
        "questions_wowpoint_tech": [
            질문1, 질문2, 질문3, ...
        ],
        "questions_wowpoint_experience": [
            질문1, 질문2, 질문3, ...
        ]
        }
        """
    @staticmethod
    def get_system_extract_doubtpoint_pf_only():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 포트폴리오 자체적으로 의심스러운 정황을 모두 파악해야 합니다.

        맥락(Context):
        * 의심스러운 정황은 경향성 측면의 의심스러운 정황이나, 논리적 모순입니다.

        지시사항(Instructions):
        * 의심스러운 정황은 반드시 포트폴리오에 드러난 내용으로만 판단해야 합니다.
        * 반드시 비어있는 기간이 6개월 이상인 경우만, 공백기에 대한 의심스러운 정황으로 판단합니다.
        * 컴퓨터 과학 분야 전공이 아님에도, 졸업 이후 개발 관련 직군 전환이 빠르게 이루어진 경우 의심스러운 정황이 될 수 있습니다.
        * JSON 형식으로 반환하세요

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"포트폴리오"},

        결과 설정(Output):
        * 형식(Format):
            'doubtsentence_pf_only' : {
                '포트폴리오 내 의심스러운 부분 1' : '의심스러운 이유',
                '포트폴리오 내 의심스러운 부분 2' : '의심스러운 이유',
                ...

            }
        """
    @staticmethod
    def get_system_extract_doubtpoint_jd_and_pf():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 주어진 포트폴리오에서 채용공고의 평가항목['자격요건', '우대사항']을 참고하여 의심스러운 정황을 모두 찾아내야 합니다.

        맥락(Context):
        * 의심스러운 정황은 경향성 측면의 의심스러운 정황이나, 논리적 모순입니다.

        지시사항(Instructions):
        * 의심스러운 정황은 반드시 포트폴리오에 드러난 내용으로만 판단해야 합니다.
        * 채용공고의 자격요건에는 존재하지만 포트폴리오에 언급되지 않은 항목의 불일치는 의심스러운 정황이 아닙니다 '절대 출력하지 마세요'.
        * 채용공고의 우대사항에는 존재하지만 포트폴리오에 언급되지 않은 항목의 불일치는 의심스러운 정황이 아닙니다 '절대 출력하지 마세요'.
        * 의심스럽지 않은 정황은 출력하지 않고, 의심스러운 정황만 출력하세요.
        * 만약 의심스러운 정황이 없다면, 빈 리스트를 반환하세요.
        * JSON 형식으로 반환하세요

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"포트폴리오"},
        채용공고 : {자격요건 :
                    ...
                    우대사항 :
                    ...},

        결과 설정(Output):
        * 형식(Format):
            doubtsentence_jd_and_pf: List[str] = Field(description="포트폴리오 내 문장")
        """
    
    @staticmethod
    def get_system_generate_questions_doubtpoint():
        return"""
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        지시사항(Instructions):
        * 주어진 문장마다 각각 면접 질문을 생성합니다.
        * 생성할 면접 질문은 주어진 포트폴리오에서 Doubt Point(의심스러운 정황) 관련 내용입니다.
        다음과 같은 Format을 참고하세요. 문장이 자연스럽지 않으면 꼭 Format을 참고하지 않아도 됩니다.
            Format :
            "A라고 이해했는데, B에서 의문점이 생깁니다. A이면 C 아닌가요?
        A는 포트폴리오에서 의심스러운 문장과 관련된 근거, B는 Doubt Point, C는 보편적인 사실을 의미합니다. (C는 자체적으로 판단하되, 구체적인 예시를 제시하세요.)
        * JSON 형식으로 반환하세요.

        제약사항(Constraints):
        * 반환 형식은 오직 JSON 형식만 반환합니다.


        입력 예시 (Input):
        * 형식(Format):
        doubtpoint_pf_only: {doubtPoint},
        doubtpoint_jd_and_pf: {doubtpoint_jd_and_pf},
        포트폴리오: {pf_original}

        결과 설정(Output):
        * 형식(Format):
        {
        "doubt_questions_pf_only": [
            질문1, 질문2, 질문3, ...
        ],
        "doubt_questions_jd_and_pf": [
            질문1, 질문2, 질문3, ...
        ]
        }
        """
    
    @staticmethod
    def get_system_extract_conformitypoint():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 주어진 포트폴리오와 채용공고의 기술키워드를 비교해 의심스러운 포트폴리오와 채용공고의 평가항목의 부합성을 찾아내야 합니다.

        맥락(Context):

        지시사항(Instructions):
        * 포트폴리오에 표면적 또는 맥락이나 의미상 입력의 '채용공고의 기술 키워드'와 부합하는 부분이 존재한다면, 그 문장을 결과의 'sentences'에 추가하세요.
        * 기술 키워드 위주로, 부합성을 판단하고 기술키워드에 존재하는 부합성 항목과 존재하지 않는 부합성 항목을 분류하세요.
        * requirements_and_preferences, is_existed_in_keywordlist '채용공고의 기술 키워드'에서 입력된 값을 그대로 사용하세요.
        * is_keywordsInPF는 해당 기술 키워드 관련 내용이 사용자 포트폴리오에 드러나 있다면 True, 아니라면 False를 반환하세요.
        * sentences는 포트폴리오 내 해당되는 문장을 모두 반환하세요.

        제약사항(Constraints):
        * 반환 형식은 오직 JSON 형식만 반환합니다.
        * 바로 파이썬 코드로 사용할 수 있게 JSON 형식에 불필요한 문자는 모두 제거하세요.
        * ```json 같은 형식은 모두 제거합니다.

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"포트폴리오"},
        채용공고의 기술 키워드 : {"채용공고의 기술 키워드"},

        결과 설정(Output):
        * 형식(Format):
            {
                "기술키워드 1": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": 'True or False,'
                    "is_keywords_in_PF" : 'True or False'
                    "sentences" : ['문장1', '문장2', ...]
                },
                "기술키워드 2": {
                    "requirements_and_preferences": 'requirements' or 'preferences',
                    "is_existed_in_keywordlist": 'True or False,
                    "is_keywords_in_PF" : 'True or False'
                    "sentences" : ['문장1', '문장2', ...]'
                },

            }
        """
    
    @staticmethod
    def get_system_generate_questions_conformitypoint_jd_and_pf():
        return """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        맥락(Context):
        * 사용자의 포트폴리오 내용을 기반으로 벡터 DB내 Similarity Search를 통한 결과물인 실제 면접 질문을 참고하여 적절한 면접 질문을 생성합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 7개 이상 생성하세요.
        * 생성되는 면접 질문은 원본을 변형하는 것이 좋습니다.
        * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]를 그대로 사용합니다.
        * 일부 질문은 사용자 포트폴리오 내의 프로젝트 명을 들며, 이 프로젝트에서는 ~ 라며 경험을 언급하는 질문도 1~2개 정도 추가하세요.
        * 사용자 포트폴리오에 언급되지 않은 기술 키워드이므로,
        1. 질문 의도는 이 사용자가 해당 기술을 사용해 봤는지에 대한 사실 확인
        2. 질문 유형은
            a. 이론적인 질문
            b. 경험적인 질문


        제약사항(Constraints):
        * 반환 형식은 오직 JSON 형식만 반환합니다.


        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"pf_original"},
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

        결과 설정(Output):
        * 형식(Format) :
            "기술 키워드 1": {
            "experience_based_questions": [
                "생성된 질문1",
                "생성된 질문2",
                ...
            ],
            "tech_based_questions": [
                "생성된 질문1",
                "생성된 질문2",
                ...
            ]
            },
            "기술 키워드 2": {
            "experience_based_questions": [
                "생성된 질문1",
                "생성된 질문2",
                ...
            ],
            "tech_based_questions": [
                "생성된 질문1",
                "생성된 질문2",
                ...
            ]
            }
            ...
        """
    
    @staticmethod
    def get_system_generate_questions_comformitypoint_jd_only():
        return """
        역할(Role):
        * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        목표(Goal):
        * 역량평가 및 면접을 위해 적절한 질문을 생성해야합니다.

        * 벡터 DB내 Similarity Search를 통한 결과물인 실제 면접 질문을 참고하여 적절한 면접 질문을 생성합니다.

        지시사항(Instructions):
        * 각 키워드별로 면접 질문을 키워드마다 각각 7개 이상 생성하세요.
        * 생성되는 면접 질문은 원본을 변형하는 것이 좋습니다.
        * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]를 그대로 사용합니다.
        * 사용자 포트폴리오에 언급되지 않은 기술 키워드이므로,
        1. 질문 의도는 이 사용자가 해당 기술을 사용해 봤는지에 대한 사실 확인
        2. 질문 유형은 경험적인 질문으로 생성하세요.
        질문이 위 유형과 적합하지 않으면 적합하도록 변경하세요. Similarity Search의 검색된 질문을 그대로 사용하지 않아도 됩니다.

        제약사항(Constraints):
        * JSON 형식만 반환합니다.
        * ```json 같은 구문 절대 사용하지 마세요.


        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {"pf_original"},
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

        결과 설정(Output):
        * 형식(Format):
            {
                "tech_keyword 1": [
                    "생성된 질문1",
                    "생성된 질문2",
                    ...
                ],
                "tech_keyword 2": []
                    "생성된 질문1",
                    "생성된 질문2",
                    ...
                ],

            }
        """
    @staticmethod
    def get_system_extract_wowpoint():
        return """
        역할(Role):
        * 당신은 다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며, 현재 기술 면접관 입니다.
        * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 면접을 담당해야 합니다.

        목표(Goal):
        * 주어진 포트폴리오와 채용공고의 평가항목['자격요건', '우대사항']을 비교해 wow-point를 찾아내야 합니다.

        맥락(Context):
        * 해당 wow-point는 포트폴리오 내 드러난 강점을 의미합니다.
        * 반환 형식은 항상 JSON 형식으로 반환하세요.

        지시사항(Instructions):
        * wow-point는 기술 범주와 경험 범주로 분류해야 합니다.
        * 기술 범주의 wow-point는 반드시 포트폴리오와 기술키워드와 관련있는 내용이되, 포트폴리오 내에 존재하는 문장이어야 합니다.
        * 기술 범주에서는 wow-point와 기술 키워드를 함께 반환하세요.

        입력 예시 (Input):
        * 형식(Format):
        포트폴리오 : {pf_original},
        채용공고 : {jd},


        결과 설정(Output):
        * 형식(Format):
        {
        "wowpoint_tech": [
            {
            "wowpoint1": "포트폴리오 내 문장 1",
            "tech_keyword": ["기술키워드1", "기술키워드2", "기술키워드3", ...],
            "reason": "wow-point로 선정한 이유",
            },
            {
            "wowpoint2": "포트폴리오 내 문장 2",
            "tech_keyword": ["기술키워드1", "기술키워드2", "기술키워드3", ...],
            "reason": "wow-point로 선정한 이유",
            },
            {
            "wowpoint3": "포트폴리오 내 문장 3",
            "tech_keyword": ["기술키워드1", "기술키워드2", "기술키워드3", ...],
            "reason": "wow-point로 선정한 이유",
            },
            ...
        ],
        "wowpoint_experience": [
            {
            "wowpoint1": "포트폴리오 내 문장 1",
            "reason": "wow-point로 선정한 이유",
            },
            {
            "wowpoint2": "포트폴리오 내 문장 2",
            "reason": "wow-point로 선정한 이유",
            },
            {
            "wowpoint3": "포트폴리오 내 문장 3",
            "reason": "wow-point로 선정한 이유",
            },
            ...
        ]
        }
        """
    

question_service_v4 = QuestionServiceV4(config=get_settings())

def get_question_service_v4() :
    yield question_service_v4