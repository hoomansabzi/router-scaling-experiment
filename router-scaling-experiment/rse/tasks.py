"""Deterministic, auto-gradable benchmark tasks.

Five categories, one required tool each:

    category   required tool      answer type
    --------   ----------------   -----------
    math       calculator         number
    units      convert_units      number
    records    lookup_record      number | name | city
    dates      date_calculator    iso_date | weekday
    text       text_tool          exact string | number

Each task also carries an `oracle_call` (the tool call that solves it). It is
used ONLY by the mock backend to check that the pipeline works, and is never
shown to a real model.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import string
from dataclasses import asdict, dataclass, field

from .kb import KB
from .tools_impl import IMPLS

CATEGORY_TOOL = {
    "math": "calculator",
    "units": "convert_units",
    "records": "lookup_record",
    "dates": "date_calculator",
    "text": "text_tool",
}
CATEGORIES = list(CATEGORY_TOOL)


@dataclass
class Task:
    task_id: str
    category: str
    question: str
    gold: str
    answer_type: str          # number | string | iso_date | weekday | name
    required_tool: str
    oracle_call: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _num(x: float) -> str:
    return repr(round(float(x), 6))


# ------------------------------- generators -------------------------------- #
def _math(rng: random.Random, i: int) -> Task:
    kind = i % 3
    if kind == 0:
        a, b, p = rng.randint(12, 98) * 100, rng.randint(37, 499), rng.choice([7, 12, 15, 18, 23, 35])
        q = (f"A warehouse stores {a} crates, each containing {b} units. {p}% of all units "
             f"are defective. How many non-defective units are there?")
        expr = f"{a}*{b}*(1-{p}/100)"
    elif kind == 1:
        a, b, c, d = rng.randint(1000, 99999), rng.randint(100, 999), rng.randint(1000, 99999), rng.randint(13, 97)
        q = f"Compute ({a} * {b} - {c}) / {d}. Give the result rounded to 3 decimal places."
        expr = f"round(({a}*{b}-{c})/{d}, 3)"
    else:
        P, r, n = rng.randint(2000, 90000), rng.choice([2.7, 3.4, 4.15, 5.25, 6.8]), rng.randint(6, 29)
        q = (f"An investment of {P} EUR grows at {r}% per year, compounded annually, for {n} "
             f"years. What is the final value in EUR, rounded to 2 decimal places?")
        expr = f"round({P}*(1+{r}/100)**{n}, 2)"
    gold = IMPLS["calculator"](expr)
    return Task(f"math-{i:03d}", "math", q, _num(float(gold)), "number", "calculator",
                {"name": "calculator", "args": {"expression": expr}})


_CONV = [
    ("furlong", "km"), ("nautical_mile", "mile"), ("fathom", "m"), ("league", "km"),
    ("chain", "yard"), ("rod", "foot"), ("stone", "kg"), ("troy_ounce", "g"),
    ("grain", "g"), ("long_ton", "kg"), ("short_ton", "tonne"), ("oil_barrel", "l"),
    ("imperial_gallon", "us_gallon"), ("us_bushel", "l"), ("cubic_foot", "us_quart"),
]


_PLURAL = {"cubic_foot": "cubic feet", "us_bushel": "US bushels", "imperial_gallon": "imperial gallons",
           "oil_barrel": "oil barrels (petroleum)", "short_ton": "short tons", "long_ton": "long tons",
           "troy_ounce": "troy ounces", "nautical_mile": "nautical miles", "stone": "stone"}
_TARGET = {"us_gallon": "US gallons", "us_quart": "US quarts", "l": "litres", "g": "grams",
           "kg": "kilograms", "km": "kilometres", "m": "metres", "tonne": "metric tonnes",
           "mile": "statute miles", "yard": "yards", "foot": "feet"}


def _units(rng: random.Random, i: int) -> Task:
    fu, tu = _CONV[i % len(_CONV)]
    v = round(rng.uniform(3, 900), rng.choice([1, 2]))
    q = (f"Convert {v} {_PLURAL.get(fu, fu + 's')} to {_TARGET.get(tu, tu)}. "
         f"Give the numeric answer to at least 4 significant figures.")
    out = IMPLS["convert_units"](v, fu, tu)
    return Task(f"units-{i:03d}", "units", q, _num(float(out.split()[0])), "number",
                "convert_units", {"name": "convert_units",
                                  "args": {"value": v, "from_unit": fu, "to_unit": tu}})


def _records(rng: random.Random, i: int) -> Task:
    rec = KB[(i * 7 + rng.randint(0, 3)) % len(KB)]
    kind = i % 4
    name = rec["project"]
    if kind == 0:
        q, gold, at = f"What is the budget in EUR of Project {name} at Veltrano Industries?", str(rec["budget_eur"]), "number"
    elif kind == 1:
        q, gold, at = f"Who is the lead of Project {name} at Veltrano Industries?", rec["lead"], "name"
    elif kind == 2:
        q, gold, at = f"In which city is Veltrano Industries' Project {name} based?", rec["office_city"], "name"
    else:
        q, gold, at = f"In what year did Veltrano Industries start Project {name}?", str(rec["start_year"]), "number"
    return Task(f"records-{i:03d}", "records", q, gold, at, "lookup_record",
                {"name": "lookup_record", "args": {"project_name": name}})


def _dates(rng: random.Random, i: int) -> Task:
    start = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 900))
    days = rng.randint(40, 700) * (1 if i % 4 else -1)
    res = start + dt.timedelta(days=days)
    word = "after" if days > 0 else "before"
    if i % 2 == 0:
        q = f"What date is {abs(days)} days {word} {start.isoformat()}? Answer in YYYY-MM-DD format."
        gold, at = res.isoformat(), "iso_date"
    else:
        q = f"What day of the week is {abs(days)} days {word} {start.isoformat()}?"
        gold, at = res.strftime("%A"), "weekday"
    return Task(f"dates-{i:03d}", "dates", q, gold, at, "date_calculator",
                {"name": "date_calculator", "args": {"start_date": start.isoformat(), "days": days}})


def _rand_word(rng, lo=6, hi=11):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def _text(rng: random.Random, i: int) -> Task:
    kind = i % 3
    if kind == 0:
        s = _rand_word(rng, 11, 16)
        q = f"Reverse the character sequence of the string '{s}'. Give only the reversed string."
        args = {"operation": "reverse", "text": s}
        at = "string"
    elif kind == 1:
        s = " ".join(_rand_word(rng, 4, 9) for _ in range(rng.randint(9, 14)))
        c = rng.choice("aeiorst")
        q = f"How many times does the letter '{c}' occur in the following text?\n\"{s}\""
        args = {"operation": "count_char", "text": s, "char": c}
        at = "number"
    else:
        s = " ".join(_rand_word(rng, 3, 8) for _ in range(rng.randint(7, 10)))
        q = f"Sort the words of this text alphabetically and return them separated by single spaces:\n\"{s}\""
        args = {"operation": "sort_words", "text": s}
        at = "string"
    gold = IMPLS["text_tool"](**args)
    return Task(f"text-{i:03d}", "text", q, gold, at, "text_tool",
                {"name": "text_tool", "args": args})


_GEN = {"math": _math, "units": _units, "records": _records, "dates": _dates, "text": _text}


def generate_tasks(n_per_category: int = 20, seed: int = 0) -> list[Task]:
    tasks = []
    for cat in CATEGORIES:
        rng = random.Random(f"{seed}-{cat}")
        tasks += [_GEN[cat](rng, i) for i in range(n_per_category)]
    return tasks


def save_tasks(tasks, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")
