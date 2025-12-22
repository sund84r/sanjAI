import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

# Absolute path to your local model folder
MODEL_DIR = "/Users/yuvaraj/Downloads/sanjai_mistral_model"

# ----------------------------
# Device selection (CUDA → MPS → CPU)
# ----------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon / AMD on macOS
else:
    DEVICE = "cpu"

# ----------------------------
# Load tokenizer & model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
# Ensure pad token exists (many causal LMs don't set it)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use float16 only on CUDA; MPS/CPU prefer float32
dtype = torch.float16 if DEVICE == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype if DEVICE == "cuda" else None,
)
model.to(DEVICE)
model.eval()

def _apply_chat_template(user_input: str) -> str:
    """
    If the tokenizer has a chat template (common for instruct-tuned Mistral),
    use it. Otherwise, fall back to a simple prompt.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_input},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback prompt
    return f"User: {user_input}\nAssistant:"

@torch.no_grad()
def generate_response(
    user_input: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05
) -> str:
    # 🔹 Apply a simple chat template if model doesn't have one
    # If you already have a chat template function, replace this
    prompt = f"<s>[INST] {user_input} [/INST]"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 🔹 Slice only the generated tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return text.strip()
