
import json, time, sys, torch
MODEL = sys.argv[1]

QA = [("What is the capital of France?", "paris"),
      ("What is 17 + 25?", "42"),
      ("What is the chemical symbol for gold?", "au"),
      ("How many legs does a spider have?", "8"),
      ("What planet is known as the Red Planet?", "mars"),
      ("What is 12 times 12?", "144"),
      ("What is the capital of Japan?", "tokyo"),
      ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
      ("What is the largest ocean on Earth?", "pacific"),
      ("What is 100 divided by 4?", "25"),
      ("Who wrote Romeo and Juliet?", "shakespeare"),
      ("What is the boiling point of water in Celsius?", "100"),
      ("What is the capital of Australia?", "canberra"),
      ("How many sides does a hexagon have?", "6"),
      ("What is the square root of 81?", "9"),
      ("What metal is liquid at room temperature?", "mercury"),
      ("What is the capital of Canada?", "ottawa"),
      ("How many minutes are in three hours?", "180"),
      ("What is the hardest natural substance?", "diamond"),
      ("What is 15 percent of 200?", "30")]

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="half", max_model_len=2048, gpu_memory_utilization=0.85)
weight_gb = torch.cuda.memory_allocated() / 1e9   # dominated by weights (KV pool grows separately)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def chat(u): return tok.apply_chat_template([{"role":"user","content":u}],
                                            add_generation_prompt=True, tokenize=False)

# 2) batch throughput
prompts = [chat(f"Describe invention number {i} that changed daily life, in about 80 words.")
           for i in range(32)]
sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
llm.generate([chat("warmup")], SamplingParams(max_tokens=8))
t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
batch_tps = sum(len(o.outputs[0].token_ids) for o in outs) / dt

# 3) interactive single-stream decode
sp1 = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
t0 = time.perf_counter(); out = llm.generate([chat("Tell me a long story about the ocean.")], sp1)
single_tps = len(out[0].outputs[0].token_ids) / (time.perf_counter() - t0)

# 4) accuracy
qs = [chat(q + " Answer with just the answer, nothing else.") for q, _ in QA]
outs = llm.generate(qs, SamplingParams(temperature=0.0, max_tokens=16))
correct = sum(a in o.outputs[0].text.lower() for o, (_, a) in zip(outs, QA))

print("RESULT " + json.dumps(dict(model=MODEL, weight_gb=round(weight_gb, 2),
      batch_tps=round(batch_tps, 1), single_tps=round(single_tps, 1),
      accuracy=f"{correct}/{len(QA)}")))
