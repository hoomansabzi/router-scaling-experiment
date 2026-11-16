"""Plain-Python implementations of the five specialist tools.

These have NO LangChain dependency, so the task generator can use them to
compute gold answers, and `tools.py` wraps them as LangChain tools.

Every task in the benchmark needs exactly one of these tools. Language models
are unreliable at these operations without a tool (big arithmetic, odd unit
conversions, calendar arithmetic, character-level string ops, and facts about
a fictional company). That makes routing *causally* matter: send the task to
an agent without the right tool and it will usually fail.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import operator
import re

from .kb import KB_BY_KEY, normalise_key

# --------------------------------------------------------------------------- #
# 1. calculator
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round":
        args = [_eval(a) for a in node.args]
        return round(*args)
    raise ValueError("unsupported expression element: " + ast.dump(node)[:60])


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Supports + - * / ** % // parentheses,
    decimals and round(x, n). Example: '(4837*127)*(1-0.15)'."""
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)  # drop thousands separators only
    expr = expr.replace("^", "**").replace("×", "*")
    value = _eval(ast.parse(expr, mode="eval"))
    if isinstance(value, float):
        value = round(value, 10)
    return str(value)


# --------------------------------------------------------------------------- #
# 2. unit converter
# --------------------------------------------------------------------------- #
_UNITS = {
    # length -> metres
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01),
    "mm": ("length", 0.001), "mile": ("length", 1609.344), "yard": ("length", 0.9144),
    "foot": ("length", 0.3048), "inch": ("length", 0.0254),
    "nautical_mile": ("length", 1852.0), "furlong": ("length", 201.168),
    "fathom": ("length", 1.8288), "league": ("length", 4828.032),
    "chain": ("length", 20.1168), "rod": ("length", 5.0292),
    # mass -> kilograms
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125), "stone": ("mass", 6.35029318),
    "troy_ounce": ("mass", 0.0311034768), "grain": ("mass", 0.00006479891),
    "short_ton": ("mass", 907.18474), "long_ton": ("mass", 1016.0469088),
    "tonne": ("mass", 1000.0),
    # volume -> litres
    "l": ("volume", 1.0), "ml": ("volume", 0.001), "us_gallon": ("volume", 3.785411784),
    "imperial_gallon": ("volume", 4.54609), "us_quart": ("volume", 0.946352946),
    "us_pint": ("volume", 0.473176473), "cubic_foot": ("volume", 28.316846592),
    "oil_barrel": ("volume", 158.987294928), "us_bushel": ("volume", 35.23907016688),
}
_ALIASES = {
    "meter": "m", "metre": "m", "meters": "m", "metres": "m", "kilometer": "km",
    "kilometre": "km", "kilometers": "km", "kilometres": "km", "miles": "mile",
    "yards": "yard", "feet": "foot", "ft": "foot", "inches": "inch", "in": "inch",
    "nautical_miles": "nautical_mile", "nmi": "nautical_mile", "furlongs": "furlong",
    "fathoms": "fathom", "leagues": "league", "chains": "chain", "rods": "rod",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g", "pound": "lb",
    "pounds": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz", "stones": "stone",
    "troy_ounces": "troy_ounce", "ozt": "troy_ounce", "grains": "grain",
    "short_tons": "short_ton", "long_tons": "long_ton", "tonnes": "tonne",
    "metric_ton": "tonne", "liter": "l", "litre": "l", "liters": "l", "litres": "l",
    "milliliter": "ml", "millilitre": "ml", "us_gallons": "us_gallon",
    "gallon": "us_gallon", "gallons": "us_gallon", "imperial_gallons": "imperial_gallon",
    "us_quarts": "us_quart", "us_pints": "us_pint", "cubic_feet": "cubic_foot",
    "barrel": "oil_barrel", "barrels": "oil_barrel", "oil_barrels": "oil_barrel",
    "us_bushels": "us_bushel", "bushel": "us_bushel",
}


def _unit(u: str) -> str:
    key = u.strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in _UNITS:
        raise ValueError(f"unknown unit '{u}'. Known: {sorted(_UNITS)}")
    return key


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a quantity between units of length, mass or volume. Unit names use
    snake_case, e.g. 'furlong', 'nautical_mile', 'troy_ounce', 'stone', 'oil_barrel',
    'imperial_gallon', 'km', 'kg', 'l'."""
    fu, tu = _unit(from_unit), _unit(to_unit)
    (fd, ff), (td, tf) = _UNITS[fu], _UNITS[tu]
    if fd != td:
        raise ValueError(f"cannot convert {fd} to {td}")
    return f"{float(value) * ff / tf:.6f} {tu}"


# --------------------------------------------------------------------------- #
# 3. fictional company records (knowledge base)
# --------------------------------------------------------------------------- #
def lookup_record(project_name: str) -> str:
    """Look up the internal record of a project at Veltrano Industries (a private
    company; this data exists nowhere else). Pass the project name, e.g. 'Halvard'
    or 'Project Halvard'. Returns lead, office city, budget in EUR and start year."""
    key = normalise_key(project_name)
    rec = KB_BY_KEY.get(key)
    if rec is None:
        close = [r["project"] for k, r in KB_BY_KEY.items() if key and (key in k or k in key)]
        if len(close) == 1:
            rec = KB_BY_KEY[normalise_key(close[0])]
        else:
            return f"NOT FOUND: no project named '{project_name}'."
    return json.dumps(rec)


# --------------------------------------------------------------------------- #
# 4. date calculator
# --------------------------------------------------------------------------- #
def date_calculator(start_date: str, days: int) -> str:
    """Add (positive) or subtract (negative) a number of days to an ISO date
    (YYYY-MM-DD). Returns the resulting ISO date and its weekday."""
    d = _dt.date.fromisoformat(start_date.strip()) + _dt.timedelta(days=int(days))
    return f"{d.isoformat()} ({d.strftime('%A')})"


# --------------------------------------------------------------------------- #
# 5. text tool
# --------------------------------------------------------------------------- #
def text_tool(operation: str, text: str, char: str = "") -> str:
    """Exact character-level text operations. operation is one of:
    'reverse' (reverse the characters of text), 'count_char' (count occurrences of
    `char` in text, case-sensitive), 'sort_words' (sort the words of text
    alphabetically, case-insensitive), 'word_count' (number of words)."""
    op = operation.strip().lower()
    if op == "reverse":
        return text[::-1]
    if op == "count_char":
        if len(char) != 1:
            raise ValueError("count_char needs a single character in `char`")
        return str(text.count(char))
    if op == "sort_words":
        return " ".join(sorted(text.split(), key=str.lower))
    if op == "word_count":
        return str(len(re.findall(r"\S+", text)))
    raise ValueError(f"unknown operation '{operation}'")


IMPLS = {
    "calculator": calculator,
    "convert_units": convert_units,
    "lookup_record": lookup_record,
    "date_calculator": date_calculator,
    "text_tool": text_tool,
}
