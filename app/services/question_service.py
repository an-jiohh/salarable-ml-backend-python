from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from app.core.config import Settings, get_settings

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# Vector Embedding
from kobert_transformers import get_kobert_model, get_tokenizer
import torch

import json
from collections import Counter

#loggin
import logging

logger = logging.getLogger(__name__)

class QuestionService:
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
        self.arrangement_pf_system_message = self.get_arrangement_pf_system_message()
        self.arrangement_jd_system_message = self.get_arrangement_jd_system_message()
        self.for_search_system_message = self.get_for_search_system_message()
        self.system_message = self.get_system_message()
        

    # def create_questions(self, id: str, links: list[str]) -> list[str]:
    def create_questions(self, portfolio_data: str, job_description_data: str, input_position: str) -> list[str]:
        
        portfolio_data, job_description_data
        pf = self.generate_response(self.arrangement_pf_system_message, portfolio_data, "list")
        jd = self.generate_response(self.arrangement_jd_system_message, job_description_data, "list")
        user_message = self.generate_userprompt_extract_sentences_from_pf(self.unique_keywords, jd, pf)
        result = self.generate_response(self.for_search_system_message, user_message, "json", "기술키워드에 해당하는 포트폴리오 내 문장들")
        keywordAndSentences_to_search = result

        query_data = {}

        for keyword in keywordAndSentences_to_search.keys():
            logger.info(f"기술키워드 : {keyword}")

            for sentence in keywordAndSentences_to_search[keyword]:
                logger.info(f"포트폴리오 내 문장 : {sentence}")
                query_results = self.query_search(keyword, sentence, input_position)
                matches = query_results.get('matches', [])
                for match in matches:
                    metadata = match.get('metadata', {})
                    tech_keyword = metadata.get('tech_keyword', 'N/A')
                    text = metadata.get('text', 'N/A')
                    score = match.get('score', 'N/A')

                    # # 출력 포맷 지정
                    # logger.info(f"Original Sentence: {sentence}")
                    # logger.info(f"Tech Keyword: {tech_keyword}")
                    # logger.info(f"Text: {text}")
                    # logger.info(f"Score: {score:.2f}")
                    # logger.info('-' * 50)

                    # 원본 문장을 키로, 검색된 텍스트 정보를 리스트로 추가
                    if sentence not in query_data:
                        query_data[sentence] = []

                    query_data[sentence].append({
                        "searched_question": text,
                        "tech_keyword": tech_keyword,
                        "score": round(score, 2)
                    })
        user_message = self.generate_userprompt_gen_question(pf, query_data)
        generated_questions=(self.generate_response(self.system_message, user_message, "json", "생성된 질문들"))

        return generated_questions
    
    def generate_userprompt_extract_sentences_from_pf(self, unique_keywords, jd, pf):
        user_message = f"""
            기술 키워드 리스트 : {unique_keywords},
            채용 공고 : {jd},
            포트폴리오 : {pf}
            """
        return user_message
    
    def query_search(self, keyword, sentence, position):

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
            namespace=position,
            vector=vectors_to_search,
            top_k=10,
            filter=filter_criteria,
            include_values=False,
            include_metadata=True
        )

        return query_results
    
    def generate_response(self, system_message, user_message, parser_type, input_description=None, params=None, kwargs=None):
            """
            사용자와 시스템 메시지를 입력 받아 응답을 생성하는 함수.

            Parameters:
            - user: 사용자 정보
            - system_message: 시스템이 제공할 메시지
            - user_message: 사용자가 입력한 메시지
            - params (dict, optional): 모델 파라미터 (temperature, max_tokens, top_p)
            - kwargs (dict, optional): 추가적인 파라미터 (frequency_penalty, presence_penalty)

            Returns:
            - mainQuestions: 생성된 질문
            """

            ### 기본 모델 파라미터 설정 ###
            default_params = {
                "temperature": 0.0,
                "max_tokens": 4096,
                "top_p": 1.0,
            }
            default_kwargs = {
                "frequency_penalty": 0,
                "presence_penalty": 0,
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

            # 메시지 출력 함수
            def print_messages(messages):
                for message in messages:
                    logger.info(f"{message.type.capitalize()} Prompt")
                    logger.info(f"Content:\n{message.content}")
                    logger.info("-" * 50)

            # 메시지 출력
            print_messages(messages)

            #smell
            class output(BaseModel):
                tech_keyword: str = Field(description=input_description)

            if parser_type == "json":
                # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
                parser = JsonOutputParser(pydantic_object=output)
                prompt = prompt.partial(format_instructions=parser.get_format_instructions())
                ### LLM Chain 생성 및 질문 생성 ###
                chain = prompt | llm | parser

            elif parser_type == "list":
                # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
                chain = prompt | llm | StrOutputParser()

            generated_response = chain.invoke({"system_input": system_message, "user_input": user_message})

            return generated_response
    
    def generate_userprompt_gen_question(self, pf, result_of_similarity_search):
            user_message = f"""
            사용자 포트폴리오: {pf},
            Result of Similarity Search: {result_of_similarity_search}
            """
            return user_message

    @staticmethod
    def init_options():
        opts = "method"
        return opts
    
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
    def get_arrangement_pf_system_message():
        return """
            역할(Role):
            * 당신은 주어진 txt형태의 포트폴리오를 보기 편하게, 전처리 해야합니다.

            목표(Goal):
            * 원본 내용을 최대한 유지하세요.
            * 포트폴리오 내의 기술 키워드들은 한글 기술키워드(영어 기술키워드)로 변환하세요. 예시: 리액트(React)
            * 포트폴리오 내 형식을 유지하기 위해 의미적으로 중요하지 않은 문장은 제거해도 됩니다.

            지시사항(Instructions):
            * 문맥의 이해에 필요없는 각종 특수기호를 제거하세요.
            * 각종 마스킹된 [link], [url], [name], ...등의 NER 태그들을 제거하세요.
            * Page 1, Page 2, ...등 Page번호를 제거하세요
        """
    @staticmethod
    def get_arrangement_jd_system_message():
        return """
            역할(Role):
            * 당신은 주어진 txt형태의 채용공고를, 전처리 해야합니다.

            목표(Goal):
            * 기술 면접을 위한 포트폴리오 요약임을 참고하세요.

            지시사항(Instructions):
            * 포트폴리오 내의 기술 키워드들은 한글 기술키워드(영어 기술키워드)로 변환하세요. 예시: 리액트(React)
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
    def get_for_search_system_message():
        return """
            역할(Role):
            * 당신은  다양한 개발 분야에서 폭넓은 경험을 가진 30년차 개발자이며 10년차 IT교육자 김철수입니다.
            * 당신은 현재 프론트엔드(FE) 및 백엔드(BE)분야의 기술 역량평가를 진행하는 평가 담당관입니다.

            목표(Goal):
            * 실제 면접 질문 데이터가 저장되어 있는 Vector Database에서 사용자의 채용공고 및 포트폴리오와 관련된 질문을 찾기 위해, 기술 키워드와 핵심 문장을 반환해야 합니다.

            맥락(Context):
            * 해당 기술이 채용공고와 포트폴리오 모두 언급된 경우, 상대적 우선도가 가장 높습니다.
            * 해당 기술이 채용공고에는 언급 되었지만, 포트폴리오에는 언급되지 않은 기술의 경우, 상대적 우선도가 중간입니다.
            * 해당 기술이 채용공고에는 언급되지 않았지만, 포트폴리오에는 언급된 경우, 상대적 우선도가 가장 낮습니다.
            * 결론적으로 채용공고의 내용이 가장 중요합니다.

            지시사항(Instructions):
            * 포트폴리오 내 문장 중 핵심 문장만 사용하세요.
            * 핵심 문장은 반드시 포트폴리오 내 존재하는 문장이어야 합니다.
            * 핵심 문장은 기술키워드가 반드시 나타나야 합니다.
            * 제시된 기술 키워드 외 다른 기술 키워드는 사용하지 마세요. 단, 영어 기술 키워드의 경우 대소문자가 달라도 철자가 같으면 상관없습니다.
            * 반환되는 기술 키워드는 제공한 기술 키워드 (영어인 경우 소문자)와 일치해야합니다.
            * JSON 형태로 결과를 반환하세요.

            제약사항(Constraints):
            * 제시된 기술 키워드 외 다른 기술 키워드는 사용하지 마세요.
            * 기술 키워드는 한글과 영어 모두 사용합니다. 예시 : 리액트(React)
            * 해당되는 문장이 없는 기술 키워드는 반환하지 않습니다.
            * 반환 형식은 오직 JSON 형식만 반환합니다.
            * 기술 키워드의 대소문자는 항상 제시한 기술 키워드 리스트와 일치해야 합니다.

            입력 예시 (Input):
            * 형식(Format):
            기술 키워드 리스트 : {"unique_keywords"},
            채용 공고 : {"채용 공고"},
            포트폴리오 : {"포트폴리오"}

            결과 설정(Output):
            * 형식(Format):
            {
                "기술 키워드1" : ["핵심 문장1", "핵심 문장2", ...],
                "기술 키워드2" : ["핵심 문장1", "핵심 문장2", ...],
                ...
            }
        """
    @staticmethod
    def get_system_message():
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
            * 각 키워드별로 면접 질문을 키워드마다 각각 7개 이상 생성하세요
            * 출력의 {tech_keyword}는 Result of Similarity Search[tech_keyword]입니다.
            * 일부 질문은 사용자 포트폴리오 내의 프로젝트 명을 들며, 이 프로젝트에서는 ~ 라며 경험을 언급하는 질문도 1~2개 정도 추가하세요.

            제약사항(Constraints):
            * 반환 형식은 오직 JSON 형식만 반환합니다.


            입력 예시 (Input):
            * 형식(Format):
            원본 포트폴리오 내용 : {"original_pf"},
            Result of Similarity Search: [{'searched_question': '검색된 질문 1', 'tech_keyword': '기술 키워드', 'score': '유사도 점수}, ...]

            결과 설정(Output):
            * 형식(Format):
            {tech_keyword} :[
                "생성된 질문1"
                "생성된 질문2",
                ...
            ],
            """

    


question_service = QuestionService(config=get_settings())

def get_question_service() :
    yield question_service