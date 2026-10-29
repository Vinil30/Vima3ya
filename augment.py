"""Label-preserving augmentation of the cleaned train set (experiment 4, "my own idea").

    python augment.py            # data/train_clean.jsonl -> data/train_aug.jsonl

For every transaction ticket we add ONE extra copy whose label stays exactly the same but whose
surface form is harder: the txn id and amount are rewritten in another valid format
("N P 1234 567 890", "Rs 1,50,000/-", "1.5 lakh", "14.4k", ...), and we sometimes add a
distracting balance/limit amount or a quoted earlier support email with its own date.
Each copy is re-checked with clean.fix_label: if the rules would change its label, it is thrown away.
"""
import json, random, re
import clean, rules

random.seed(0)
AMT = re.compile(r"(?:(?:₹|\brs\.?|\binr)\s?)?(\d[\d,]*(?:\.\d+)?)(\s?(?:k|lakhs?)\b)?(\s(?:rupees|rupaye))?(/-)?", re.I)


def indian(n):                      # 150000 -> 1,50,000
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return head + "," + tail


def amount_text(paise):
    rup, p = divmod(paise, 100)
    num = f"{rup}.{p:02d}" if p else str(rup)
    styles = [f"Rs. {num}", f"₹ {num}", f"INR {indian(rup)}" + (f".{p:02d}" if p else ""),
              f"{num} rupees", f"Rs {indian(rup)}/-", f"₹{num}" + ("" if p else ".00")]
    if not p and rup >= 1000 and rup % 100 == 0:
        styles.append(f"{rup / 1000:g}k")
    if not p and rup >= 100000 and rup % 1000 == 0:
        styles.append(f"{rup / 100000:g} lakh")
    return random.choice(styles)


def id_text(tid):
    d = tid[2:]
    return random.choice([f"np{d}", f"NP {d[:5]} {d[5:]}", f"NP-{d}", f"N P {d[:4]} {d[4:7]} {d[7:]}",
                          f"np {d[:5]} {d[5:]}", f"NP {d}"])


def augment(user, lab):
    head, text = user.split("\n\n", 1)
    if lab["txn_id"]:
        m = re.search(r"\bn\s*p[\s-]*((?:\d[\s-]?){9}\d)(?!\d)", text, re.I)
        if m:
            text = text[:m.start()] + id_text(lab["txn_id"]) + text[m.end():]
    if lab["amount_paise"]:
        for m in AMT.finditer(text):
            v = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").strip().lower()
            v *= 1000 if unit == "k" else 100000 if unit.startswith("lakh") else 1
            if round(v * 100) == lab["amount_paise"]:
                text = text[:m.start()] + amount_text(lab["amount_paise"]) + text[m.end():]
                break
    if random.random() < 0.5:       # distracting amount that must be ignored
        x = amount_text(random.randint(1, 3000) * 100 * random.choice([1, 10, 100]))
        s = (random.choice([f"Account mein abhi {x} pade hain.", f"Meri daily limit {x} hai."]) if lab["language"] == "hi-en"
             else random.choice([f"My account balance is {x} right now.", f"FYI my daily limit is {x}."]))
        cuts = [m.end() for m in re.finditer(r"[.!?] ", text)] or [len(text)]
        c = random.choice(cuts)
        text = text[:c] + s + " " + text[c:]
    if "Source: email" in head and random.random() < 0.5:   # quoted earlier email with its own date
        d = rules.received(user).toordinal() - random.randint(2, 40)
        d = __import__("datetime").date.fromordinal(d).strftime("%d %b %Y")
        text += (f"\n\n> On {d}, NimbusPay Support wrote:\n> Thank you for contacting NimbusPay. "
                 f"Your request has been registered with ticket #{random.randint(10000, 99999)}.")
    return head + "\n\n" + text


def main():
    rows = clean.load("data/train_clean.jsonl")
    out = list(rows)
    for r in rows:
        lab = json.loads(r["messages"][2]["content"])
        if lab["category"] not in clean.TXN_CATS:
            continue
        user = augment(r["messages"][1]["content"], lab)
        new, reasons = clean.fix_label(dict(lab), user)
        if reasons or new != lab or (lab["amount_paise"] and lab["amount_paise"] not in clean.amounts_in(rules.body(user))):
            continue                                    # augmentation changed the meaning -> skip
        msgs = [r["messages"][0], {"role": "user", "content": user}, r["messages"][2]]
        out.append({"id": r["id"] + "-aug", "messages": msgs})
    random.shuffle(out)
    with open("data/train_aug.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clean rows + {len(out) - len(rows)} augmented = {len(out)}")


if __name__ == "__main__":
    main()
