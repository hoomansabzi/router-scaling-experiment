"""A small, deterministic, *fictional* knowledge base.

Facts about 'Veltrano Industries' do not exist anywhere on the internet, so a
model cannot answer these questions from its training data: only the agent
that owns the `lookup_record` tool can.
"""
from __future__ import annotations

import random
import re

_CODENAMES = [
    "Halvard", "Quillon", "Marisol", "Tessaro", "Brennick", "Ozymel", "Calderon",
    "Vintra", "Lumeral", "Sorrento", "Kestrel", "Abernai", "Frostline", "Ombrelle",
    "Pyrite", "Zenobia", "Galloway", "Ithaca", "Corvane", "Rubicel", "Nadirex",
    "Solenne", "Thistle", "Varuna", "Wexford", "Yarrow", "Elandor", "Moravia",
    "Petrichor", "Lindqvist", "Saffron", "Haldane", "Orrery", "Juniper", "Castellan",
    "Brightwater", "Draxis", "Fenwick", "Isolde", "Quintara",
]
_FIRST = ["Ilaria", "Tomasz", "Renske", "Joaquim", "Saoirse", "Dmitri", "Amaia", "Kofi",
          "Liesel", "Ravi", "Noor", "Elio", "Maren", "Teodor", "Yusra", "Bastian",
          "Oona", "Matteo", "Ingrid", "Farid"]
_LAST = ["Vasquelle", "Ondrasek", "Haverkamp", "Lindholm", "Okonkwo-Reyes", "Petrakis",
         "Szabados", "Moreau-Blanc", "Achterberg", "Castellanos", "Nyberg", "Ferraro",
         "Kowalczyk", "Delacroix", "Bjornsdottir", "Zanetti", "Quaresma", "Halloran"]
_CITIES = ["Lisbon", "Tallinn", "Ljubljana", "Gdansk", "Porto", "Bergen", "Graz",
           "Brno", "Valencia", "Turku", "Ghent", "Bologna", "Cluj-Napoca", "Aarhus"]


def _build(seed: int = 1234):
    rng = random.Random(seed)
    records = []
    for name in _CODENAMES:
        records.append({
            "project": name,
            "company": "Veltrano Industries",
            "lead": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "office_city": rng.choice(_CITIES),
            "budget_eur": rng.randrange(120_000, 9_800_000, 1_000),
            "start_year": rng.randint(2009, 2026),
        })
    return records


def normalise_key(s: str) -> str:
    s = s.lower().replace("project", "")
    return re.sub(r"[^a-z]", "", s)


KB = _build()
KB_BY_KEY = {normalise_key(r["project"]): r for r in KB}
