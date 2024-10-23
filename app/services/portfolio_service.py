from app.core.config import Settings, get_settings
from collections import defaultdict
import re
import fitz
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class PortfolioService :
    def __init__(self, config:Settings) -> None:
        self.email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        self.phone_pattern = r"\b(?:010|011|016|017|018|019|10)(?:\s*\-\s*|\s+)\d{3,4}(?:\s*\-\s*|\s+)\d{4}\b"
        self.edu_pattern = r"\b\S+대학교\b|\b\S+학교\b"
        tlds = (
            r"com|net|org|edu|gov|mil|int|co|us|uk|de|jp|fr|au|ca|cn|br|in|ru|pl|it|nl|se|no|fi|dk|ch|at|be|es|pt|gr|hk|kr|sg|tw|my|ph|za|mx|ar|cl|pe|uy|do|pa|cr|gt|hn|sv|jm|tt|ky|ai|lc|vc|ms|ws|io"
        )
        self.url_pattern = re.compile(r"""
            (?:http[s]?://)?                   # http:// 또는 https:// (선택 사항)
            (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+  # 도메인 이름
            (?:""" + tlds + r""")            # TLD
            (?:\:[0-9]{1,5})?                  # 포트 번호 (선택 사항)
            (?:/[a-zA-Z0-9-._~:/?#@!$&'()*+,;=%]*)?  # URL 경로, 쿼리 문자열 및 프래그먼트 (선택 사항)
            """, re.VERBOSE)
    
    def mask_portfolio(self, pdf_path: str, search_texts: list, replace_texts: list):
        doc = fitz.open(pdf_path)
        temp_images = []
        substitution_counts = defaultdict(int)

        logger.info(search_texts)
        logger.info(replace_texts)

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            self._mask_basic_info(text, search_texts, replace_texts, page, substitution_counts)

            for search_text, replace_text in zip(search_texts, replace_texts):
                text_instances = page.search_for(search_text)
                for inst in text_instances:
                    x0, y0, x1, y1 = inst
                    page.draw_rect(inst, color=(1, 1, 1), fill=(1, 1, 1))
                    page.insert_text((x0, (y0 + y1) // 2), f"[{replace_text}]", fontsize=8, fontname="helv")
                    substitution_counts[f"{search_text} -> {replace_text}"] += 1
                
            temp_image_path = self._save_page_as_image(page, page_num)
            temp_images.append(temp_image_path)

        output_pdf_path = self._generate_masked_pdf(pdf_path, temp_images)

        doc.close()

        return output_pdf_path



    def _mask_basic_info(self, text, search_texts, replace_texts, page, substitution_counts):
        """기본 텍스트 정보 (이메일, 전화번호 등) 마스킹 처리"""
        email_addresses = re.findall(self.email_pattern, text)
        for email in email_addresses:
            if email not in search_texts:
                search_texts.append(email)
                replace_texts.append("[email]")

        phone_numbers = re.findall(self.phone_pattern, text)
        for phone in phone_numbers:
            if phone not in search_texts:
                search_texts.append(phone)
                replace_texts.append("[phone]")

        edus = re.findall(self.edu_pattern, text)
        for edu in edus:
            if edu not in search_texts:
                search_texts.append(edu)
                replace_texts.append("[edu]")

        urls = re.findall(self.url_pattern, text)
        for url in urls:
            if url not in search_texts:
                search_texts.append(url)
                replace_texts.append("[url]")

    def _generate_masked_pdf(self, pdf_path, image_list):
        """저장된 이미지 파일들을 합쳐서 새로운 마스킹된 PDF 생성"""
        path_without_extension = os.path.splitext(pdf_path)[0]
        output_path = f"{path_without_extension}_마스킹.pdf"

        images = [Image.open(img_path) for img_path in image_list]
        images[0].save(output_path, save_all=True, append_images=images[1:])

        #최적화 필요
        for img_path in image_list:
            if os.path.exists(img_path):
                os.remove(img_path)

        return output_path
    
    def _save_page_as_image(self, page, page_num):
        """PDF 페이지를 이미지로 변환하여 임시 파일로 저장"""
        dpi = 300
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_image_path = f"temp_image/page_{page_num + 1}.png"
        img.save(temp_image_path, quality=95)
        return temp_image_path



portfolio_service = PortfolioService(config=get_settings())

def get_portfolio_service() :
    yield portfolio_service

