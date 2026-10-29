# NimbusPay triage: report

**Setup.** `Qwen/Qwen2.5-3B-Instruct`, QLoRA (4-bit NF4 base model + LoRA r=16 on all linear layers, about 30M trainable params), Hugging Face `transformers`/`peft`/`bitsandbytes` with the plain `Trainer`, free Colab T4.

I chose plain HF over Unsloth or Axolotl so that the chat template, loss mask and collator are each a few explicit lines I can explain. I chose Qwen2.5-3B because it is the strongest instruct model under 4B that fits a T4 with 2.3k-token examples, and it has a native system role.

## A. Data audit (`clean.py`, `cleaning_log.csv`)

**Method.** I turned every rule in `SCHEMA.md` into a check (`rules.py`) and ran the checks on train **and on dev**. Dev is clean, so it is the control: every check flags **0 dev rows**, and the fixer changes **0 of 200 dev labels**. That shows the checks encode the spec and not my guesses. I then read flagged rows by hand to name each problem.

| # | Problem found | How I found it | Rows | Action |
|---|---|---|---|---|
| 1 | Exact or near duplicates (same text up to case, whitespace, "Thx") | normalised-text hash | 100 | drop (keep first) |
| 2 | **Leakage:** ticket also in dev | same hash against dev/test | 45 | drop |
| 3 | Label is truncated, invalid JSON | `json.loads` fails | 25 | drop |
| 4 | Junk tickets ("test", "asdf asdf", empty) with made-up labels | very short body | 10 | drop |
| 5 | **Old spec v1:** key `prio` with `high/medium/low`, dates as `DD/MM/YYYY` | key set ≠ the 8 spec keys | 70 | fix: rename key, recompute priority, ISO dates |
| 6 | Priority contradicts the P1/P2/P3 rules (e.g. `offers` marked P1, fraud marked P3) | recompute from category, amount, legal words | 60 | fix: recompute |
| 7 | `amount_paise` actually in rupees (e.g. `46094` for "46094 rupaye") | label not among amounts in text, but label × 100 is | 25 | fix: × 100, keeping the paise from the text |
| 8 | `txn_id` not normalised (`NP-…`, `np …`, `np 12345 67890`) | regex `NP\d{10}` | 21 | fix: uppercase, strip separators |

In total: **180 dropped, 175 fixed, 1,265 rows kept**.

Fixing amounts (7) also changed priority on some rows, for example ₹46,094 makes P3 into P2. Checked and found clean: `needs_human` (always consistent with the true priority and the repeat-contact phrases), transaction fields on non-transaction categories (all `null`), `channel`, `language`, `category`. For category, a leave-one-out Naive Bayes and a fraud-keyword cross-check flagged rows, but on reading them they were correct.

## B–D. Results on dev (200 rows, `score.py`)

All fine-tuned runs differ from `clean_qlora` in exactly **one** setting.

| Run | What changed | Exact match | Mean field acc | Peak GPU (GB) | Train min |
|---|---|---|---|---|---|
| baseline_zero_shot | base model, fixed prompt | 0.0 % | 0.1 % | 2.90 | 0 |
| baseline_rules_in_prompt | base model + SCHEMA.md in prompt | 12.0 % | 77.9 % | 3.55 | 0 |
| clean_qlora | reference: clean data, 4-bit, r16, 2 ep, lr 2e-4 | 88.5 % | 97.9 % | 12.03 | 23.1 |
| raw_qlora | (1) raw data (only dev/test leaks removed) | 88.0 % | 97.9 % | 12.03 | 24.7 |
| clean_lora16 | (2) 16-bit LoRA instead of 4-bit QLoRA (batch 1 × accum 16: the fp16 model OOMs at batch 2; same effective batch) | 84.5 % | 97.0 % | 10.02 | 27.0 |
| clean_1ep | (3) 1 epoch instead of 2 | 56.5 % | 92.2 % | 12.02 | 11.3 |
| **aug_qlora** (final) | (4) + hard surface-form augmentation | **93.5 %** | **98.8 %** | 12.20 | 39.3 |

Per field (dev): category 99.5/98.5 (aug / clean, %), priority 95.5/92.5, txn_date 98.0/97.0, needs_human 97.5/95.5, txn_id 100/99.5; amount, channel and language are 100 % for both. The zero-shot base model scores 0 % because it answers in prose or with its own keys. Putting the rules in the prompt fixes the format but still gets priority right only 35.5 % of the time.

**Pipeline choices (Part C).**
- Prompt = system + user through `tok.apply_chat_template(add_generation_prompt=True)`, which is exactly the inference format.
- Loss only on the answer JSON + `<|im_end|>`, so the model learns to stop. Prompt tokens are labelled −100. The notebook prints the decoded loss tokens as proof.
- MAX_LEN 2304, from measured lengths: p95 is about 250 tokens, but quoted email threads reach 2,255. I keep them because they teach "ignore quoted dates". Nothing is truncated.
- Batch 2 × accumulation 8, so long threads fit on the T4.
- lr 2e-4 cosine, 2 epochs, fp16 (the T4 has no bf16), paged 8-bit AdamW, gradient checkpointing, `group_by_length`.

**Raw data experiment.** I removed only the 45 dev-leak rows, because training on them would be training on dev.

**Experiment 4 (my idea).** The test set is "harder than dev". `SCHEMA.md` lists the traps: other amounts, quoted dates, and id formats. So for each transaction ticket I add one copy with:
- the id and amount rewritten in another valid format (`N P 1234 567 890`, `Rs 1,50,000/-`, `1.5 lakh`, `14.4k`);
- sometimes a distracting balance or limit amount;
- for emails, a quoted support reply with its own date.

Labels are unchanged, and every copy is re-checked by the rule fixer, which discards it if its label would change. This adds 835 rows.

**Interpretation.** Note that 200 dev rows give a standard error of about ±2.3 points on exact match, and each run used one seed.
- **Training length mattered most.** With 1 epoch, the model learns the format but not the rules: priority accuracy drops to 63.5 % and exact match to 56.5 %.
- **Augmentation was the best addition** (+5.0 exact match, about 2 standard errors). It mainly improved priority and needs_human, the fields that depend on reading the amount correctly.
- **Cleaning made almost no difference on dev** (88.5 vs 88.0). The rule violations were a minority of rows, and with 2 epochs the model learns the majority rule anyway. The fixes matter more for consistency than for this score.
- **16-bit LoRA was not better than 4-bit QLoRA** (84.5 vs 88.5). It had to run at batch 1, so this is not a pure precision comparison. Its peak memory is lower only because of that smaller batch. QLoRA lets the T4 use batch 2 with long email threads, and it lost no accuracy.

## E. Error analysis (final model `aug_qlora`, dev)

There are 19 wrong fields in 13 tickets (187/200 exact):
- priority: 9
- needs_human: 5
- txn_date: 4
- category: 1

By category, the errors are in refund (8 fields), payment_failed (8), other (2) and fraud (1). account_access, kyc and offers are 100 %.

**Main finding: priority errors are amount-threshold errors.** 8 of the 9 priority errors are rupee amounts near a cutoff, and the model puts them on the wrong side:
- P1 starts at ₹50,000, i.e. 5,000,000 paise.
- P2 starts at ₹5,000 for payment_failed or refund.

The model extracts the amount correctly (100 % amount accuracy), but it compares the amount to the cutoff poorly. **All 5 needs_human errors follow from a wrong priority**, because P1 implies needs_human. So the model applies the needs_human rule correctly, and the root cause is the threshold comparison.

| # | id | field: gold → pred | Why |
|---|---|---|---|
| 1 | dv-00129 | priority P2 → P1 (+ needs_human) | Rs. 49,657 is just under ₹50,000; the model rounds it up to the P1 band |
| 2 | dv-00148 | priority P2 → P1 | INR 42,062 is under ₹50,000. A "second complaint" plus 16 quoted replies make the ticket look urgent |
| 3 | dv-00056 | priority P2 → P1 (+ needs_human) | Rs 44,206.75: the paise part makes the number look bigger |
| 4 | dv-00098 | priority P2 → P1 (+ needs_human) | "41.5k" is under ₹50,000. A ₹1,871 balance is also mentioned as a distractor |
| 5 | dv-00034 | priority P2 → P1 (+ needs_human) | Hinglish "Rs. 36737 ka transaction": the amount is read correctly but put in the wrong band |
| 6 | dv-00011 | priority P1 → P2 (+ needs_human) | ₹84,137 is above ₹50,000, so this is P1. "URGENT!!" alone is not a P1 signal, and the model also misses the amount |
| 7 | dv-00044 | priority P2 → P3 | Refund of Rs. 8,377 ≥ ₹5,000 should be P2 |
| 8 | dv-00127 | priority P3 → P2 | ₹4,836 is just under the ₹5,000 P2 cutoff. A ₹111 balance is also mentioned |
| 9 | dv-00084 | category other → account_access (+ priority) | "I didn't get a PIN yet" sounds like an access problem, but it is a card or PIN delivery issue. A wrong category changes priority (account_access is P2) |
| 10 | dv-00006 | txn_date 06-30 → 07-01 | "day before yesterday" on 2 July crosses a month boundary. The model subtracts 1 day instead of 2 |
| 11 | dv-00171 | txn_date 05-24 → 05-26 | The date is "24-05-26" (DD-MM-YY). The model takes the year digits "26" as the day |
| 12 | dv-00087 | txn_date 08-05 → null | "4 days ago" is at the end of a long email with a "new card" story and a distractor balance. The model does not find the relative date |
| 13 | dv-00114 | txn_date 08-30 → null | "Dated Aug 30" (no year) comes after a ticket number "#75402" and a legal-notice sentence. The model misses it |

**Next week.**
- **Threshold augmentation.** Make copies of tickets with amounts just above and just below ₹5,000 and ₹50,000, written in every format (k, lakh, paise, Hinglish). This directly targets the 14 of 19 wrong fields that are priority or needs_human.
- **Date augmentation.** Add relative dates across month and year boundaries, DD-MM-YY dates whose year looks like a day, and dates buried at the end of long emails.
- **Test on more seeds** (3 seeds per run) so that differences smaller than 3 points can be trusted.
- **Try r=32 or 3 epochs.** Going from 1 to 2 epochs helped a lot, so more capacity or more training may still help. Watch dev for overfitting.

## AI tools used

AI Tools Used

I used Claude Code (Anthropic) as a coding assistant to support development, while I led the project’s technical decisions, implementation, experimentation, and evaluation.

# My role:

Defined the project objectives and constraints, including QLoRA fine-tuning, code simplicity, and maximising the development score.

Selected and refined the technical approach, making decisions based on the project requirements.

Implemented and adapted the proposed solutions, using AI assistance where useful.

Ran experiments on Google Colab, debugged runtime issues, including CUDA 13 / bitsandbytes incompatibility, out-of-memory errors during 16-bit training, and session disconnections.

Evaluated notebook outputs, verified the results, and made decisions based on experimental findings.

Reviewed and finalised the report to accurately reflect the methodology and results.

# Claude Code’s role:

Assisted with exploring model choices and experiment configurations.

Suggested potential data augmentation approaches and implementation improvements.

Helped generate and refine code for rules.py, clean.py, augment.py, and the training notebook.

Provided assistance with debugging and report drafting.

All reported metrics are based on actual notebook outputs. No AI model was used to label the data or generate test predictions; predictions were produced using the fine-tuned adapter.
