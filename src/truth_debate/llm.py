from __future__ import annotations

from pathlib import Path
from typing import Any


class HFGenerator:
    def __init__(
        self,
        model_cfg: dict[str, Any],
        adapter_path: str | Path | None = None,
        trainable_lora: bool = False,
        lora_cfg: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_cfg = model_cfg
        self.model_name = str(model_cfg["name"])
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            )
        except OSError as exc:
            raise RuntimeError(_model_load_help(self.model_name)) from exc
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _resolve_dtype(torch, str(model_cfg.get("dtype", "auto")))
        quantization_config = None
        if bool(model_cfg.get("load_in_4bit", False)):
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype if dtype is not None else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            )
        except OSError as exc:
            raise RuntimeError(_model_load_help(self.model_name)) from exc

        if trainable_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if bool(model_cfg.get("load_in_4bit", False)):
                self.model = prepare_model_for_kbit_training(self.model)
            if not lora_cfg:
                raise ValueError("lora_cfg is required when trainable_lora=True")
            peft_cfg = LoraConfig(
                r=int(lora_cfg["r"]),
                lora_alpha=int(lora_cfg["alpha"]),
                lora_dropout=float(lora_cfg["dropout"]),
                target_modules=list(lora_cfg["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg)
            self.model.print_trainable_parameters()
        elif adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

        self.model.train(trainable_lora)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered: list[str] = []
            for msg in messages:
                rendered.append(f"{msg['role'].upper()}:\n{msg['content']}")
            rendered.append("ASSISTANT:\n")
            return "\n\n".join(rendered)

    def generate(
        self,
        messages_or_prompt: list[dict[str, str]] | str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        prompt = self.format_messages(messages_or_prompt) if isinstance(messages_or_prompt, list) else messages_or_prompt
        max_new_tokens = int(max_new_tokens or self.model_cfg.get("max_new_tokens", 128))
        temperature = float(self.model_cfg.get("temperature", 0.7) if temperature is None else temperature)
        top_p = float(self.model_cfg.get("top_p", 0.9) if top_p is None else top_p)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        do_sample = temperature > 0

        was_training = self.model.training
        self.model.eval()
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if was_training:
            self.model.train()
        new_tokens = output[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def sequence_logprob(self, prompt: str, completion: str) -> tuple[Any, int]:
        text = prompt + completion
        full = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full = {key: value.to(self.device) for key, value in full.items()}

        labels = full["input_ids"].clone()
        prompt_len = min(prompt_ids.shape[-1], labels.shape[-1])
        labels[:, :prompt_len] = -100
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens == 0:
            return self.torch.tensor(0.0, device=self.device), 0
        outputs = self.model(**full, labels=labels)
        logprob_sum = -outputs.loss * valid_tokens
        return logprob_sum, valid_tokens

    def save_adapter(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)


def _resolve_dtype(torch, dtype_name: str):
    if dtype_name == "auto":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def _model_load_help(model_name: str) -> str:
    return (
        f"Could not load model '{model_name}'. If this happened on Vast.ai, the instance/container "
        "probably cannot reach huggingface.co or the model is not cached. Run "
        "`truth-debate preflight --config <config>` to verify networking. If networking is unavailable, "
        "download the model on a machine with internet, copy it to the instance, and set `model.name` "
        "in the config to that local directory."
    )
