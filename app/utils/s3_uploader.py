import boto3
import os
import logging
from fastapi import HTTPException
from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self, config:Settings):
        self.bucket_name = config.s3_bucket_name
        self.s3_client = boto3.client('s3')

    def upload_file(self, file_path: str, object_name: str = None) -> str:
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            s3_url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
            logger.info(f"파일이 성공적으로 업로드되었습니다: {s3_url}")
            return s3_url
        except Exception as e:
            logger.error(f"{file_path}를 S3에 업로드 실패: {str(e)}")
            raise HTTPException(status_code=500, detail="S3 업로드 실패")

    def delete_file(self, object_name: str):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"S3에서 파일이 삭제되었습니다: {object_name}")
        except Exception as e:
            logger.error(f"S3에서 파일 삭제 실패: {str(e)}")
            raise HTTPException(status_code=500, detail="S3 파일 삭제 실패")
        
s3_uploader = S3Uploader(config=get_settings())

def get_s3_uploader() :
    yield s3_uploader