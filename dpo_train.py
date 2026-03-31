"""
Direct Preference Optimization (DPO) — Super Simple Version
============================================================

WHAT IS DPO?
------------
DPO aligns a language model with human preferences WITHOUT needing a
separate reward model or reinforcement learning. It's a simpler
alternative to RLHF (Reinforcement Learning from Human Feedback).

THE KEY IDEA (in plain English):
--------------------------------
1. You have PAIRS of responses to the same prompt:
   - A "chosen" response (the one humans preferred)
   - A "rejected" response (the one humans didn't prefer)

2. DPO directly adjusts the model's weights so it becomes MORE likely
   to produce "chosen"-style outputs and LESS likely to produce
   "rejected"-style outputs.

3. The math trick: DPO reformulates the RL objective into a simple
   classification loss. Instead of:
     Train reward model -> Run RL with PPO -> Align model
   You just do:
     Train model directly on preference pairs

THE DPO LOSS (intuition):
--------------------------
  loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

  where log_ratio = log(policy(response) / reference(response))

  In English: "Make the model prefer chosen over rejected responses,
  relative to how the original (reference) model behaved."

  beta controls how far the model can drift from the reference model.
  Higher beta = stay closer to original model.
"""

import json
import os
import time
import torch

os.environ["WANDB_DISABLED"] = "true"
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


# ============================================================
# Metrics collector — saves training data for the web dashboard
# ============================================================
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.logs = []
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            entry["step"] = state.global_step
            entry["elapsed_sec"] = round(time.time() - self.start_time, 1)
            self.logs.append(entry)


metrics_cb = MetricsCallback()
results = {
    "config": {},
    "training_data": [],
    "training_logs": [],
    "test_results": [],
    "status": "starting",
}


# ============================================================
# STEP 1: Load a small model (we use a tiny one for demo)
# ============================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

USE_CUDA = torch.cuda.is_available()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if not USE_CUDA else torch.float16,
    device_map="auto" if USE_CUDA else None,
)

if not USE_CUDA:
    model = model.to("cpu")


# ============================================================
# STEP 2: Create preference data
# ============================================================
PREFERENCE_DATA = [
    {
        "prompt": "Explain what a black hole is.",
        "chosen": "A black hole is a region in space where gravity is so "
                  "strong that nothing, not even light, can escape. They form "
                  "when massive stars collapse at the end of their life cycle.",
        "rejected": "A black hole is a hole that is black. It sucks things in. "
                     "Nobody really knows what they are.",
    },
    {
        "prompt": "How do I make scrambled eggs?",
        "chosen": "Crack 2-3 eggs into a bowl, whisk with salt and pepper. "
                  "Heat butter in a pan over medium-low heat, pour in eggs, "
                  "and gently stir with a spatula until softly set.",
        "rejected": "Put eggs in pan. Cook them. Add stuff if you want.",
    },
    {
        "prompt": "What is machine learning?",
        "chosen": "Machine learning is a branch of AI where computers learn "
                  "patterns from data instead of being explicitly programmed. "
                  "For example, a spam filter learns from labeled emails.",
        "rejected": "Machine learning is when computers learn stuff. "
                     "It's really complicated and uses lots of math.",
    },
    {
        "prompt": "Why is exercise important?",
        "chosen": "Regular exercise strengthens your heart, improves mood by "
                  "releasing endorphins, helps maintain a healthy weight, and "
                  "reduces the risk of chronic diseases like diabetes.",
        "rejected": "Exercise is good for you. You should do it because "
                     "everyone says so.",
    },
    {
        "prompt": "Explain recursion in programming.",
        "chosen": "Recursion is when a function calls itself to solve smaller "
                  "sub-problems. For example, factorial(5) = 5 * factorial(4). "
                  "Every recursive function needs a base case to stop.",
        "rejected": "Recursion is a hard concept. It's when things repeat. "
                     "You'll understand it eventually.",
    },
]

results["training_data"] = PREFERENCE_DATA


def format_as_chat(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def build_dataset():
    rows = []
    for ex in PREFERENCE_DATA:
        rows.append({
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "chosen": format_as_chat(ex["prompt"], ex["chosen"]),
            "rejected": format_as_chat(ex["prompt"], ex["rejected"]),
        })
    return Dataset.from_list(rows)


dataset = build_dataset()
print(f"Dataset size: {len(dataset)} preference pairs")


# ============================================================
# STEP 3: Configure LoRA
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# STEP 4: Configure DPO training
# ============================================================
training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    warmup_steps=10,
    logging_steps=1,
    fp16=USE_CUDA,
    gradient_checkpointing=True,
    do_eval=False,
    remove_unused_columns=False,
)

results["config"] = {
    "model": MODEL_NAME,
    "beta": training_args.beta,
    "epochs": int(training_args.num_train_epochs),
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "dataset_size": len(dataset),
}


# ============================================================
# STEP 5: Train with DPO!
# ============================================================
print("\nStarting DPO training...")
results["status"] = "training"

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
    callbacks=[metrics_cb],
)

trainer.train()
results["training_logs"] = metrics_cb.logs
results["status"] = "generating"
print("\nTraining complete!")


# ============================================================
# STEP 6: Test the aligned model
# ============================================================
print("\nGenerating test responses...")

test_prompts = [
    "What is gravity?",
    "How do I learn Python?",
    "What makes a good friend?",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results["test_results"].append({"prompt": prompt, "response": response.strip()})
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.strip()}")


# ============================================================
# STEP 7: Save everything
# ============================================================
trainer.save_model("./dpo_output/final")
tokenizer.save_pretrained("./dpo_output/final")

results["status"] = "complete"
with open("./dpo_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nResults saved to dpo_results.json")
print("Model saved to ./dpo_output/final")
