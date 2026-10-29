"""Audit + clean data/train.jsonl  ->  data/train_clean.jsonl and cleaning_log.csv

    python clean.py

Every problem found in the audit is handled here. Each dropped or fixed row is written to
cleaning_log.csv with the reason(s). Uses only train.jsonl and SCHEMA.md rules; dev/test are
read only to detect leaked (overlapping) tickets, never to produce labels.
"""
import csv, json, re
import rules

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
TXN_CATS = {"payment_failed", "refund", "fraud", "offers"}
JUNK = {"", "test", "test test 123", "asdf asdf"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(user):
    """Ticket text normalised for duplicate detection (case, whitespace, trailing 'thx')."""
    return re.sub(r"\s+", " ", re.sub(r"\b(thx|thanks)\b", "", user.lower())).strip(" .")


def amounts_in(text):
    """All amounts written in the text, in paise (handles commas, k, lakh)."""
    out = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(k|lakhs?)?\b", text.replace(",", ""), re.I):
        v, unit = float(m.group(1)), (m.group(2) or "").lower()
        v *= 1000 if unit == "k" else 100000 if unit.startswith("lakh") else 1
        out.add(round(v * 100))
    return out


def fix_label(lab, user):
    """Return (fixed_label, [reasons]). Applies the SCHEMA.md rules to one label."""
    reasons, text = [], rules.body(user)

    if "prio" in lab:                                   # old spec: key 'prio' with high/medium/low
        lab["priority"] = lab.pop("prio")
        reasons.append("old-spec key 'prio' (high/medium/low) -> 'priority'")
    lab = {k: lab.get(k) for k in KEYS}                 # fixed key order

    if lab["category"] in TXN_CATS:
        amt = lab["amount_paise"]
        if amt is not None and amt not in amounts_in(text):          # amount stored in rupees, not paise
            match = [a for a in amounts_in(text) if a // 100 == amt]
            if match:
                lab["amount_paise"] = match[0]
                reasons.append(f"amount in rupees not paise: {amt} -> {match[0]}")
        d = rules.find_date(user)
        d = d.isoformat() if d else None
        if d != lab["txn_date"]:
            reasons.append(f"txn_date {lab['txn_date']} -> {d}")
            lab["txn_date"] = d
        t = rules.txn_id(user)
        if t != lab["txn_id"]:
            reasons.append(f"txn_id {lab['txn_id']} -> {t}")
            lab["txn_id"] = t
    else:                                               # schema: always null for these categories
        for k in ("amount_paise", "txn_date", "txn_id", "channel"):
            if lab[k] is not None:
                reasons.append(f"{k} must be null for {lab['category']}")
                lab[k] = None

    p = rules.priority(lab["category"], lab["amount_paise"], text)
    if p != lab["priority"]:
        reasons.append(f"priority {lab['priority']} -> {p}")
        lab["priority"] = p
    nh = rules.needs_human(p, text)
    if nh != lab["needs_human"]:
        reasons.append(f"needs_human {lab['needs_human']} -> {nh}")
        lab["needs_human"] = nh
    return lab, reasons


def main():
    train = load("data/train.jsonl")
    held_out = {norm(r["messages"][1]["content"]) for f in ("data/dev.jsonl", "data/test_inputs.jsonl") for r in load(f)}
    seen, kept, log = set(), [], []

    for r in train:
        user, raw = r["messages"][1]["content"], r["messages"][2]["content"]
        key = norm(user)
        drop = None
        if rules.body(user).strip().lower() in JUNK:
            drop = "junk ticket text (test/asdf/empty) with a made-up label"
        elif key in held_out:
            drop = "same ticket appears in dev/test (leakage)"
        elif key in seen:
            drop = "duplicate of an earlier train ticket"
        else:
            try:
                lab = json.loads(raw)
            except json.JSONDecodeError:
                drop = "label is truncated / invalid JSON"
        if drop:
            log.append((r["id"], "drop", drop))
            continue
        seen.add(key)
        lab, reasons = fix_label(lab, user)
        if reasons:
            log.append((r["id"], "fix", "; ".join(reasons)))
        r["messages"][2]["content"] = json.dumps(lab, ensure_ascii=False)
        kept.append(r)

    with open("data/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(log)
    print(f"kept {len(kept)} / {len(train)} rows; dropped {sum(a == 'drop' for _, a, _ in log)}, "
          f"fixed {sum(a == 'fix' for _, a, _ in log)}")


if __name__ == "__main__":
    main()
