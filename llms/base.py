from abc import ABC, abstractmethod

class LLM(ABC):
  
    @abstractmethod
    def generate_content(self, prompt: str):
        """Generate content with given prompt"""