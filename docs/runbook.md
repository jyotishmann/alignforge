# AlignForge Runbook

*How to reproduce the results from scratch.*

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or 3.12 |
| CUDA GPU | ≥ 16GB VRAM (T4, A10G, RTX 3090) — or free Colab T4 |
| Storage | ≥ 30GB free (datasets + models + checkpoints) |
| Ollama | Latest (for serving) |
| uv | Latest |

---

## Step 1: Environment setup

```bash
git clone https://github.com/<your-username>/alignforge
cd alignforge
make setup   # creates .venv, installs dev+serve+ui extras
source .venv/bin/activate
alignforge doctor   # verify environment
```

On Colab: open `notebooks/alignforge_colab.ipynb', "Run all each session". Cell 1 handles setup.

---

## Step 2: Build the eval suites (once, before any training)

```bash
# Install data extras.
uv pip install -e ".[data]"

# Build MT-Bench and AlpacaEval subsets.
python scripts/build_eval_suites.py

# Write your 50 domain cases (see Part 07 authoring guide).
# Edit evals/domain_v1.jsonl — must have 50 entries before eval.
```

---

## Step 3: Build the dataset

```bash
# Full build (~30 min, downloads HF datasets).
alignforge data build \
  --config configs/data/sft_dev_assistant.yaml

# Note the content hash in the output, e.g. "d7c2f0aabbcc".
# Then build the preference dataset:
alignforge data build \
  --config configs/data/dpo_dev_assistant.yaml

# Note its hash too.
```

Debug mode (fast, no HF download needed for pipeline testing):

```bash
alignforge data build --limit 100 --skip-embedding
```

---

## Step 4: SFT training

```bash
# Dry run (verify config, no model load).
alignforge train sft \
  --config configs/train/sft_qlora.yaml \
  --model-config configs/model/qwen2_5_1_5b.yaml \
  --dataset-hash <sft-hash-from-step-3> \
  --dry-run

# Real run (3–4 hours on T4).
alignforge train sft \
  --config configs/train/sft_qlora.yaml \
  --model-config configs/model/qwen2_5_1_5b.yaml \
  --dataset-hash <sft-hash>
  # On Colab, add: --drive-sync /content/drive/MyDrive/alignforge
```

Note the `run_id` from the output (e.g., `sft-20250312-9c1e`).

---

## Step 5: DPO alignment

```bash
alignforge train dpo \
  --config configs/train/dpo_qlora.yaml \
  --model-config configs/model/qwen2_5_1_5b.yaml \
  --sft-run <sft-run-id> \
  --pref-hash <pref-hash-from-step-3>

# Optional: β sweep (3 runs, takes 3× as long).
alignforge train dpo-sweep \
  --sft-run <sft-run-id> \
  --pref-hash <pref-hash> \
  --betas 0.05,0.1,0.3 \
  --limit 500
```

Note the DPO `run_id`.

---

## Step 6: Register models for evaluation

```bash
# Register the base model (no adapter — points to model name).
alignforge registry publish \
  --run <sft-run-id> --as base \
  --display "Base (Qwen2.5-1.5B)" \
  --backend transformers \
  --weights Qwen/Qwen2.5-1.5B-Instruct \
  --sort 1

alignforge registry publish \
  --run <sft-run-id> --as sft \
  --display "SFT" --backend transformers --sort 2

alignforge registry publish \
  --run <dpo-run-id> --as dpo \
  --display "DPO (β=0.1)" --backend transformers --sort 3
```

---

## Step 7: Evaluation

```bash
# Generation (~45 min on T4 for all 3 models × 150 cases).
alignforge eval generate --models base,sft,dpo

# Judging (~$0.50 at gpt-4o-mini rates for 150 cases × 3 pairs × 2 orderings).
# Set ALIGNFORGE_JUDGE_API_KEY in your .env first.
alignforge eval judge --models base,sft,dpo --backend openai

# Metrics + report.
alignforge eval report --models base,sft,dpo
```

The report is written to `reports/eval_<id>.md` and `.html`.
Commit it: `git add reports/ && git commit -m "exp(eval): ..."`.

---

## Step 8: Export to GGUF

```bash
# Build llama.cpp (once per machine, ~4 min).
bash scripts/setup_llama_cpp.sh

# Export.
alignforge export gguf \
  --dpo-run <dpo-run-id> \
  --config configs/export/gguf_q4km.yaml

# Register for Ollama serving.
alignforge registry publish \
  --run <dpo-run-id> --as dpo \
  --display "DPO (β=0.1)" \
  --backend ollama --sort 3
```

---

## Step 9: Serve

```bash
# Requires Ollama running: ollama serve &
alignforge serve all --engine ollama
# → UI:  http://localhost:7860
# → API: http://localhost:8000/docs
```

---

## Reproduction checksum

| Run  | Config hash | Dataset hash | Win rate   |
| ---- | ----------- | ------------ | ---------- |
| SFT  | `<fill in>` | `<fill in>`  | —          |
| DPO  | `<fill in>` | `<fill in>`  | 0.68       |
| Eval | `<eval id>` | —            | see report |

*Replace with your actual hashes after running.*

---

## Troubleshooting

| Symptom                  | Cause                    | Fix                                                                                   |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------------------- |
| OOM at step N            | Batch size too large     | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`          |
| Loss stuck at 0.0        | All labels masked        | Response template mismatch — check `get_response_template()` output                   |
| DPO KL exploding         | LR too high or β too low | Use `configs/train/dpo_qlora.yaml` defaults                                           |
| GGUF generates gibberish | Chat template mismatch   | Run parity test: `pytest tests/integration/test_chat_format_parity.py -m integration` |
| UI shows no models       | Registry not seeded      | Run `alignforge registry publish` or `python scripts/seed_example_db.py`              |
| `curl` hangs             | Ollama not running       | `ollama serve &`                                                                      |
