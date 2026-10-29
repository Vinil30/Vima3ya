# NimbusPay ticket triage: QLoRA fine-tune

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Loaded in 4-bit?** | **Yes.** bitsandbytes NF4, double quantisation, fp16 compute dtype (same at train and inference) |
| **Adapter** | `adapter/`: PEFT LoRA, r=16, alpha=32, all 7 linear projections, ~30M trainable params |
| **Best result (dev)** | `aug_qlora`: exact match **93.5 %**, mean field accuracy **98.8 %** on the 200 dev tickets. The base model with the rules in the prompt scores 12.0 % / 77.9 % |

## Files

| File | What |
|---|---|
| `adapter/` | Final LoRA adapter (`aug_qlora`) in PEFT format. `adapter_model.safetensors` is stored with Git LFS |
| `notebook.ipynb` | The whole pipeline with outputs: audit → baselines → training → experiments → test predictions → error analysis |
| `rules.py` | The `SCHEMA.md` rules as small functions. Used **only** to audit and fix training labels, never at inference |
| `clean.py` | Writes `data/train_clean.jsonl` and `cleaning_log.csv` |
| `augment.py` | Label-preserving "hard surface form" augmentation (experiment 4) |
| `cleaning_log.csv` | Every dropped or fixed train row with its reason |
| `predictions_test.jsonl` | Raw model outputs for the 400 test tickets |
| `report.md` | Findings, experiment table, error analysis |
| `score.py`, `SCHEMA.md` | Provided with the assignment, unchanged (the notebook uses them) |

The training, dev and test data are not in this repository. To rerun the notebook, put `train.jsonl`, `dev.jsonl` and `test_inputs.jsonl` in `data/`.

## How to load the adapter and predict

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", quantization_config=bnb,
                                            torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()

ids = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True, return_tensors="pt").to(0)
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
```

Test predictions were generated one ticket at a time, as above. Qwen2.5 has a native system role, so the system message is passed as-is. The model's own `generation_config` is left untouched.

Library versions: `transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 bitsandbytes>=0.48` (Colab now ships CUDA 13; older bitsandbytes has no CUDA 13 build), free Colab T4.
