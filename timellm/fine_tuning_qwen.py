from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model, tokenizer = FastLanguageModel. from_pretrained(
    model_name="Qwen/Qwen3-14B",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# config lora finetuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)