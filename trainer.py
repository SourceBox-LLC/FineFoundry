import unsloth
from unsloth import FastLanguageModel
from datasets import Dataset
import os
import sys
import time
import math
import logging
import torch
import json

# ---- Logging setup (colored if rich is available) ----
try:
    from rich.logging import RichHandler
    _USE_RICH = True
except Exception:
    _USE_RICH = False

def _setup_logging():
    if _USE_RICH:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("trainer")

log = _setup_logging()

# Make Hugging Face logs flow through our logger cleanly
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()
    hf_logging.disable_default_handler()
except Exception:
    pass

# Simple ANSI colors for emphasis (works in Docker/Linux)
class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    INFO = "\033[96m"
    BAD = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

def cfmt(color, text):
    return f"{color}{text}{C.END}"

def log_gpu(prefix=""):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        props = torch.cuda.get_device_properties(dev)
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = props.total_memory / (1024**3)
        logging.info(f"{prefix}GPU[{dev}] {name} | alloc={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB")
    else:
        logging.info(f"{prefix}CPU mode (CUDA not available)")

# Environment snapshot
log.info(cfmt(C.BOLD, "=== Environment ==="))
log.info("torch=%s | cuda=%s | devices=%s", torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.device_count())
log_gpu("INIT ")

# Load model
max_seq_length = 2048  # We auto support RoPE Scaling internally
dtype = None  # None for auto detection. Float16 for T4/V100, Bfloat16 for Ampere+
load_in_4bit = True  # 4bit quantization reduces memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)




file = json.load(open("scraped_training_data.json", "r"))
log.info(cfmt(C.BOLD, "=== Data ==="))
log.info("Loaded %d records from scraped_training_data.json", len(file))

def format_prompt(example):
    return f"### Input: {example['input']}\n### Output: {json.dumps(example['output'])}"

formatted_data = [format_prompt(item) for item in file]
dataset = Dataset.from_dict({"text": formatted_data})
log.info("Dataset built: %d examples", len(dataset))

##############################
# TRAINING
##############################
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

class HeartbeatCallback(TrainerCallback):
    def __init__(self, every_steps: int = 25, every_seconds: int = 60):
        self.every_steps = every_steps
        self.every_seconds = every_seconds
        self._last = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        log.info(cfmt(C.BOLD, "Training started"))
        log_gpu("BEGIN ")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else "?"
        log.info(cfmt(C.INFO, f"Epoch {epoch} begin"))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        do_log = False
        if self.every_steps and state.global_step % self.every_steps == 0:
            do_log = True
        if self.every_seconds and (now - self._last) >= self.every_seconds:
            do_log = True
        if do_log:
            logs = kwargs.get("logs") or {}
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if loss is not None:
                log.info("Step %s | loss=%.4f | lr=%s", state.global_step, loss, lr)
            else:
                log.info("Step %s", state.global_step)
            log_gpu("STEP  ")
            self._last = now
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else "?"
        log.info(cfmt(C.INFO, f"Epoch {epoch} end"))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        log.info(cfmt(C.BOLD, "Training finished"))
        log_gpu("END   ")
        return control

# Training arguments optimized for Unsloth
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    callbacks=[HeartbeatCallback(every_steps=25, every_seconds=60)],
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=25,
        disable_tqdm=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=False,
        report_to="none", # Disable Weights & Biases logging
    ),
)

# Effective batch size and steps/epoch estimation
world_size = int(os.environ.get("WORLD_SIZE", "1"))
eff_bs = 2 * 4 * world_size  # per_device_train_batch_size * grad_accum * world_size
steps_per_epoch = math.ceil(len(dataset) / eff_bs) if eff_bs > 0 else None
log.info(cfmt(C.BOLD, "=== Training Plan ==="))
log.info("world_size=%s | effective_batch_size=%s | steps_per_epoch≈%s", world_size, eff_bs, steps_per_epoch)

# show memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
log.info("GPU = %s | Max memory = %.3f GB", gpu_stats.name, max_memory)
log.info("Start reserved memory = %.3f GB", start_gpu_memory)

# training
log.info(cfmt(C.BOLD, "=== Train ==="))
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
log.info("train_runtime: %.2fs (%.2f min)", trainer_stats.metrics["train_runtime"], round(trainer_stats.metrics["train_runtime"]/60, 2))
log.info("Peak reserved memory = %.3f GB (training delta = %.3f GB)", used_memory, used_memory_for_lora)
log.info("Peak reserved mem %s of max = %.3f %% (training %% = %.3f %%)", "%", used_percentage, lora_percentage)

# Test the fine-tuned model
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Test prompt
messages = [
    {"role": "user", "content": "someone is bullying me. what should I do?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# Decode and print
response = tokenizer.batch_decode(outputs)[0]
log.info(cfmt(C.OK, "=== Sample Generation ==="))
log.info(response)