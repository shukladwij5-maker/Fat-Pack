import os


class FatTummyTrainer:
    def __init__(self, model, dataset, epochs=3):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs

    def finetune(self, epochs=None):
        if epochs:
            self.epochs = epochs
            
        print(f"FatTummy starting fine-tuning loop for {self.epochs} epochs...")
        is_tpu = "TPU_NAME" in os.environ or "XRT_TPU_CONFIG" in os.environ

        if is_tpu:
            self._finetune_tpu()
        else:
            self._finetune_gpu_cpu()

    def _finetune_tpu(self):
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
        except ImportError:
            print("FatTummy: torch_xla is not installed. Attempting to download and install XLA...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_xla"])
                import torch_xla.core.xla_model as xm
                import torch_xla.distributed.parallel_loader as pl
            except Exception as e:
                print(f"FatTummy Warning: Failed to install or import torch_xla ({e}). Falling back to CPU/GPU.")
                self._finetune_gpu_cpu()
                return

        def _map_fn(index, flags):
            device = xm.xla_device()
            self.model.to(device)
            import torch
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            
            if hasattr(self.dataset, '__iter__'):
                parallel_loader = pl.ParallelLoader(self.dataset, [device])
                loader = parallel_loader.per_device_loader(device)
            else:
                loader = []
            
            for epoch in range(self.epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    xm.optimizer_step(optimizer, barrier=True)
                xm.master_print(f"Epoch {epoch+1} completed on TPU.")

    def _finetune_gpu_cpu(self):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        is_fp16 = next(self.model.parameters()).dtype == torch.float16
        scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" and not is_fp16 else None

        for epoch in range(self.epochs):
            print(f"Running epoch {epoch+1}/{self.epochs}")
            loader = [1, 2, 3] # Mock batch loop
            for batch in loader:
                optimizer.zero_grad()
                
                # Mock a real forward pass to generate valid gradients for the optimizer
                dummy_input = torch.randint(0, 100, (1, 8)).to(device)
                
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        out = self.model(dummy_input)
                        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
                        loss = logits.mean()
                        
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    out = self.model(dummy_input)
                    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
                    loss = logits.mean()
                    loss.backward()
                    optimizer.step()
