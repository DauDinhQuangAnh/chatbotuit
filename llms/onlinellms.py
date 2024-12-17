from constant import GEMINI
import google.generativeai as genai
from .base import LLM
import backoff

class OnlineLLMs(LLM):
    def __init__(self, name, api_key=None, model_version=None):
        self.name = name  
        self.model = None  
        self.model_version = model_version  

        if self.name.lower() == GEMINI and api_key:
            genai.configure(
                api_key=api_key  
            )
            self.model = genai.GenerativeModel(
                model_name=model_version  
            )

    def set_model(self, model):
        self.model = model  

    def parse_message(self, messages):
        mapping = {
            "user": "user",  
            "assistant": "model"  
        }
        # Duyệt qua danh sách tin nhắn và định dạng lại vai trò và nội dung
        return [
            {"role": mapping[mess["role"]], "parts": mess["content"]}
            for mess in messages
        ]
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
        # Kiểm tra nếu tên mô hình là Gemini
        if self.name.lower() == GEMINI:
            try:
                messages = self.parse_message(messages)  # Chuyển đổi vai trò của các tin nhắn
                response = self.model.generate_content(
                    [
                        {"role": "user", "parts": system_prompt},  # Tin nhắn của hệ thống
                        {"role": "model", "parts": "I understand. I will strictly follow your instruction!"},
                        *messages  # Thêm các tin nhắn đã xử lý
                    ],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,  # Giới hạn số token tối đa
                        temperature=temperature  # Độ ngẫu nhiên trong phản hồi
                    )
                )
                return response.text  # Trả về nội dung phản hồi của mô hình
            except Exception as e:
                print(f"Error occurred: {e}, retrying...")  # Thông báo lỗi và thử lại
                raise e   
        else:
            raise ValueError(f"Unknown model name: {self.name}")  # Lỗi nếu tên mô hình không đúng

    def generate_content(self, prompt):
        """Sinh nội dung dựa trên mô hình LLM được chỉ định."""
        if not self.model:
            raise ValueError("Model is not set. Please set a model using set_model().")

        # Sinh nội dung dựa trên loại mô hình
        if self.name.lower() == GEMINI:
            # Xử lý cấu trúc phản hồi của Gemini
            response = self.model.generate_content(prompt)
            try:
                # Trích xuất văn bản từ cấu trúc phản hồi lồng nhau
                content = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse the Gemini response structure.")
        else:
            raise ValueError(f"Unknown model name: {self.name}")
        if not isinstance(content, str):
            content = str(content)

        return content  
