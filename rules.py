"""SCHEMA.md rules as code. Used ONLY to audit and fix train labels, never at inference."""
import re
import datetime as dt

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                           "jul", "aug", "sep", "oct", "nov", "dec"])}
LEGAL = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.I)
REPEAT = re.compile(r"third time|second complaint|written \d+ times|contacted support about this before|"
                    r"again and again|teesri baar|pehle bhi complaint|baar baar", re.I)


def body(user):
    """Customer text: drop the Received/Source header and quoted '>' lines."""
    b = user.split("\n\n", 1)[1]
    return "\n".join(l for l in b.split("\n") if not l.startswith(">"))


def received(user):
    return dt.date.fromisoformat(user[10:20])


def priority(cat, amt, text):
    if cat == "fraud" or (amt or 0) >= 5_000_000 or LEGAL.search(text):
        return "P1"
    if (cat in ("payment_failed", "refund") and (amt or 0) >= 500_000) or cat == "account_access":
        return "P2"
    return "P3"


def needs_human(prio, text):
    return prio == "P1" or bool(REPEAT.search(text))


def _latest(rec, mo, d):
    """Most recent date with this month/day on or before the received date."""
    for y in (rec.year, rec.year - 1):
        try:
            x = dt.date(y, mo, d)
        except ValueError:
            continue
        if x <= rec:
            return x


def find_date(user):
    rec, t = received(user), body(user).lower()
    for pat, n in [(r"day before yesterday|\bparso\b", 2), (r"\byesterday\b|\bkal\b", 1), (r"\btoday\b|\baaj\b", 0)]:
        if re.search(pat, t):
            return rec - dt.timedelta(days=n)
    m = re.search(r"(\d+)\s*(days? ago|din pehle)", t)
    if m:
        return rec - dt.timedelta(days=int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", t)
    if m:
        d, mo, y = map(int, m.groups())
        return dt.date(y + 2000 if y < 100 else y, mo, d)
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3})[a-z]*\.?(?:\s+(20\d\d))?\b", t):
        if m.group(2) in MONTHS:
            d, mo = int(m.group(1)), MONTHS[m.group(2)]
            return dt.date(int(m.group(3)), mo, d) if m.group(3) else _latest(rec, mo, d)
    for m in re.finditer(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d))?", t):
        if m.group(1) in MONTHS:
            mo, d = MONTHS[m.group(1)], int(m.group(2))
            return dt.date(int(m.group(3)), mo, d) if m.group(3) else _latest(rec, mo, d)
    return None


def txn_id(user):
    m = re.search(r"\bn\s*p[\s-]*((?:\d[\s-]?){9}\d)(?!\d)", body(user), re.I)
    return "NP" + re.sub(r"\D", "", m.group(1)) if m else None
