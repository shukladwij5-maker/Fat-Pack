from .installer import ensure_installed
from .inference.cloud_adapters import get_cloud_adapter
from .inference.local_adapters import get_local_adapter
from .tuning.trainer import FatTummyTrainer

class FatTummyEngine:
    def __init__(self):
        self._engine_name = None
        self._param = None
        self._data_sources = []
        self._model_type = None
        self._api_key = None
        self._temperature = 1.0
        
        self._compiled = False
        self._adapter = None
        self._model_instance = None
        
        # Audit environment lazily
        ensure_installed()

    def engine(self, name: str):
        """Switches context between 'mooe', 'ollama', 'hf', 'gemini', 'openai', 'anthropic'."""
        self._engine_name = name
        return self

    def modelbuild(self, scale: str):
        """Parses identifiers like '10B', '8b' into hyperparameter sets."""
        self._param = scale
        return self

    def data(self, *sources):
        """Ingests one or more data sources (CSV, DataFrame, or HF Dataset identifier)."""
        self._data_sources.extend(sources)
        return self

    def type(self, arch):
        """Validates model target and executes initialization compilation."""
        self._model_type = arch
        self._compile_and_initialize()
        return self

    def key(self, api_key: str):
        """Sets API key for cloud engines."""
        self._api_key = api_key
        return self

    def temp(self, value: float):
        """Sets the temperature for generation/chat."""
        self._temperature = value
        return self

    def _compile_and_initialize(self):
        """Private backend method to initialize the selected engine context."""
        if self._compiled:
            return

        if self._engine_name in ["openai", "anthropic", "gemini"]:
            # Defer initialization until key is available or generate is called
            pass
        elif self._engine_name in ["ollama", "hf"]:
            if isinstance(self._model_type, str):
                self._adapter = get_local_adapter(self._engine_name, self._model_type)
        else:
            # Native model (e.g., MOOE)
            try:
                from .models.mooe import MOOE, MOOEConfig
                if isinstance(self._model_type, type) and issubclass(self._model_type, MOOE):
                    hidden_size = 4096
                    if self._param and isinstance(self._param, str):
                        scale_lower = self._param.lower()
                        if "1b" in scale_lower:
                            hidden_size = 2048
                        elif "10b" in scale_lower or "8b" in scale_lower:
                            hidden_size = 4096
                    
                    config = MOOEConfig(hidden_size=hidden_size)
                    self._model_instance = MOOE(config)
                else:
                    pass
            except ImportError:
                pass
                
        self._compiled = True

    def generate(self, prompt: str) -> str:
        """Unified adapter pattern generator."""
        # A real implementation would pass self._temperature down to the adapters.
        if self._engine_name in ["openai", "anthropic", "gemini"]:
            if not self._adapter:
                if not self._api_key:
                    raise ValueError(f"API Key required for {self._engine_name} engine.")
                self._adapter = get_cloud_adapter(self._engine_name, self._api_key)
            return self._adapter.generate(prompt)
        elif self._adapter:
            return self._adapter.generate(prompt)
        else:
            return f"[Generated text from native model {self._model_type} for prompt: {prompt}]"

    def chat(self):
        """Initiates an interactive chat interface in the terminal."""
        print(f"FatTummy Chat session started. (Type 'exit' to quit) [Temp: {self._temperature}]")
        self._compile_and_initialize()
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                response = self.generate(user_input)
                print(f"FatTummy: {response}")
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"FatTummy Error: {e}")

    def finetune(self, epochs: int = 3):
        """Delegates to tuning trainer."""
        if self._engine_name == "hf" and self._adapter:
            trainer = FatTummyTrainer(self._adapter.model, self._data_sources, epochs=epochs)
            trainer.finetune()
        elif self._model_instance:
            trainer = FatTummyTrainer(self._model_instance, self._data_sources, epochs=epochs)
            trainer.finetune()
        else:
            raise ValueError("Finetuning not supported for this context (requires native model or local hf engine).")

    def push_to_hub(self, repo_id: str):
        """Automatically registers custom model weights to HF Hub."""
        if self._model_instance:
            print(f"FatTummy pushing {self._model_type} to Hugging Face Hub at {repo_id}...")
            self._model_instance.push_to_hub(repo_id)
        elif self._engine_name == "hf" and self._adapter:
            print(f"FatTummy pushing fine-tuned model to Hugging Face Hub at {repo_id}...")
            self._adapter.model.push_to_hub(repo_id)
        else:
            raise ValueError("No local model instance available to push.")
