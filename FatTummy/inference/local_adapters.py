import subprocess
from ..exceptions import FatTummyNetworkError, FatTummyOOMError

class LocalAdapterBase:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OllamaAdapter(LocalAdapterBase):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._ensure_model_pulled()

    def _ensure_model_pulled(self):
        # Programmatically verify if ollama has the model downloaded
        try:
            # Check list of models
            output = subprocess.check_output(["ollama", "list"]).decode("utf-8")
            if self.model_name not in output:
                print(f"FatTummy: Model '{self.model_name}' not found locally. Pulling via Ollama...")
                subprocess.check_call(["ollama", "pull", self.model_name])
        except subprocess.CalledProcessError as e:
            print(f"FatTummy Warning: Failed to interface with Ollama daemon: {e}")

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except Exception as e:
            raise FatTummyNetworkError("ollama_generate", original_error=str(e))

class HuggingFaceAdapter(LocalAdapterBase):
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype=torch.float16
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise FatTummyOOMError(original_error=str(e))
            raise

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        try:
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise FatTummyOOMError(original_error=str(e))
            raise

def get_local_adapter(engine_name: str, model_name: str):
    if engine_name == "ollama":
        return OllamaAdapter(model_name)
    elif engine_name == "hf":
        return HuggingFaceAdapter(model_name)
    else:
        raise ValueError(f"Unknown local engine: {engine_name}")
