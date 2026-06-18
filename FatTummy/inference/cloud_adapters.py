import os
from ..exceptions import FatTummyNetworkError

class CloudAdapterBase:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OpenAIAdapter(CloudAdapterBase):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        
    def generate(self, prompt: str, model: str = "gpt-4o") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise FatTummyNetworkError(operation="openai_generate", original_error=str(e))

class AnthropicAdapter(CloudAdapterBase):
    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def generate(self, prompt: str, model: str = "claude-3-opus-20240229") -> str:
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise FatTummyNetworkError(operation="anthropic_generate", original_error=str(e))

class GeminiAdapter(CloudAdapterBase):
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        
    def generate(self, prompt: str, model: str = "gemini-2.5-pro") -> str:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            raise FatTummyNetworkError(operation="gemini_generate", original_error=str(e))

def get_cloud_adapter(engine_name: str, api_key: str):
    if engine_name == "openai":
        return OpenAIAdapter(api_key)
    elif engine_name == "anthropic":
        return AnthropicAdapter(api_key)
    elif engine_name == "gemini":
        return GeminiAdapter(api_key)
    else:
        raise ValueError(f"Unknown cloud engine: {engine_name}")
