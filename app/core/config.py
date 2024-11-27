from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    pinecone_api_key: str 
    openai_api_key: str
    pinecone_index_name: str
    pre_processed_dataset: str
    s3_bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    rtzr_client_id: str
    rtzr_client_secret: str
    rtzr_api_url: str

    # APP_ENV: str = 'dev'
    # vscode에서 python env 설정으로 필요없어짐
    # model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings():
    return Settings()