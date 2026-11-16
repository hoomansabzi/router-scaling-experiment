"""Agent definitions and the agent *pools* that make up each experimental condition.

An agent = name + description (what the router sees) + set of tool names.

Pool types (the independent variable is the number of agents N):

  generalist   N=1  one agent that owns all five tools, no router (baseline).
  granularity  N=2,3,5  the five tools split over fewer/more specialists.
               Coverage is always complete, so a perfect router loses nothing.
  irrelevant   N=5+k  the five core specialists + k agents from unrelated
               domains (translator, poet, ...) with no tools.
  decoy        N=5+k  the five core specialists + k agents whose descriptions
               OVERLAP the task domains (finance analyst, calendar assistant,
               HR assistant...) but who do NOT own the needed tool.

A route is counted as correct iff the chosen agent owns the task's required tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    tools: tuple = field(default_factory=tuple)
    kind: str = "core"


CORE = [
    Agent("math_agent", "Solves arithmetic and numeric word problems exactly using a calculator "
          "(percentages, compound interest, multi-step arithmetic).", ("calculator",)),
    Agent("unit_agent", "Converts physical quantities between units of length, mass and volume "
          "(including imperial and unusual units like furlongs, stone, troy ounces).", ("convert_units",)),
    Agent("records_agent", "Has access to the internal project database of Veltrano Industries: "
          "project leads, office cities, budgets and start years.", ("lookup_record",)),
    Agent("date_agent", "Does calendar arithmetic: adds or subtracts days from dates and finds "
          "the weekday of a date.", ("date_calculator",)),
    Agent("text_agent", "Performs exact character-level string operations: reversing strings, "
          "counting letters, sorting words, counting words.", ("text_tool",)),
]

MERGED = {
    2: [
        Agent("quant_agent", "Handles anything numeric: exact arithmetic and word problems, unit "
              "conversions (length, mass, volume) and calendar/date arithmetic.",
              ("calculator", "convert_units", "date_calculator"), "merged"),
        Agent("info_agent", "Handles information and text: the Veltrano Industries internal project "
              "database (leads, cities, budgets, start years) and exact string operations "
              "(reverse, count letters, sort words).", ("lookup_record", "text_tool"), "merged"),
    ],
    3: [
        Agent("quant_agent", "Exact arithmetic and numeric word problems, plus unit conversions "
              "of length, mass and volume.", ("calculator", "convert_units"), "merged"),
        Agent("records_agent", CORE[2].description, ("lookup_record",), "merged"),
        Agent("time_text_agent", "Calendar arithmetic (adding/subtracting days, weekdays) and exact "
              "string operations (reverse, count letters, sort words).",
              ("date_calculator", "text_tool"), "merged"),
    ],
}

IRRELEVANT = [
    Agent("translator_agent", "Translates text between natural languages such as English, Italian, German and French.", (), "irrelevant"),
    Agent("poetry_agent", "Writes poems, song lyrics and creative verse on any theme.", (), "irrelevant"),
    Agent("sql_agent", "Writes and explains SQL queries for relational databases.", (), "irrelevant"),
    Agent("email_agent", "Drafts professional e-mails and business letters.", (), "irrelevant"),
    Agent("legal_agent", "Reviews contracts and explains legal clauses in plain language.", (), "irrelevant"),
    Agent("recipe_agent", "Suggests recipes and meal plans for given ingredients and diets.", (), "irrelevant"),
    Agent("travel_agent", "Recommends travel destinations, sights and itineraries.", (), "irrelevant"),
    Agent("sentiment_agent", "Classifies the sentiment and tone of customer reviews.", (), "irrelevant"),
    Agent("code_review_agent", "Reviews Python code for bugs, style and readability.", (), "irrelevant"),
    Agent("fitness_agent", "Designs workout routines and explains exercises.", (), "irrelevant"),
    Agent("music_agent", "Recommends music and explains music theory concepts.", (), "irrelevant"),
]

DECOY = [
    Agent("finance_agent", "Financial analyst: answers questions about money, investments, interest, "
          "budgets and percentages.", (), "decoy"),
    Agent("engineering_agent", "Engineering assistant: physical quantities, measurements, weights and "
          "dimensions of objects.", (), "decoy"),
    Agent("hr_agent", "HR assistant: answers questions about employees, team leads, offices and "
          "company organisation.", (), "decoy"),
    Agent("calendar_agent", "Calendar assistant: schedules, deadlines, meetings and dates.", (), "decoy"),
    Agent("copyedit_agent", "Copy editor: edits and proofreads text, fixes spelling and word order.", (), "decoy"),
    Agent("statistics_agent", "Statistician: computes numbers, averages, totals and growth rates.", (), "decoy"),
    Agent("logistics_agent", "Logistics planner: warehouses, shipments, cargo weights, distances and volumes.", (), "decoy"),
    Agent("company_wiki_agent", "Company knowledge assistant: general information about companies, "
          "their projects and history.", (), "decoy"),
    Agent("project_manager_agent", "Project manager assistant: project timelines, start dates, "
          "budgets and staffing.", (), "decoy"),
    Agent("linguist_agent", "Linguist: analyses words, letters, spelling and the structure of text.", (), "decoy"),
    Agent("science_tutor_agent", "Science tutor: physics and chemistry problems, units and measurements.", (), "decoy"),
]

GENERALIST = Agent("generalist_agent", "A single agent with every tool.",
                   ("calculator", "convert_units", "lookup_record", "date_calculator", "text_tool"),
                   "generalist")


def build_pool(pool_type: str, n_agents: int) -> list[Agent]:
    if pool_type == "generalist":
        assert n_agents == 1
        return [GENERALIST]
    if pool_type == "granularity":
        if n_agents == 5:
            return list(CORE)
        if n_agents in MERGED:
            return list(MERGED[n_agents])
        raise ValueError("granularity pools exist for N = 2, 3, 5")
    extra = n_agents - len(CORE)
    if extra < 0:
        raise ValueError("irrelevant/decoy pools need N >= 5")
    src = {"irrelevant": IRRELEVANT, "decoy": DECOY}[pool_type]
    if extra > len(src):
        raise ValueError(f"at most N={len(CORE) + len(src)} for {pool_type}")
    return list(CORE) + list(src[:extra])


# Default sweep. N=5 is the shared anchor of the three curves.
DEFAULT_CONDITIONS = [
    ("generalist", 1),
    ("granularity", 2), ("granularity", 3), ("granularity", 5),
    ("irrelevant", 8), ("irrelevant", 12), ("irrelevant", 16),
    ("decoy", 8), ("decoy", 12), ("decoy", 16),
]


def owners(pool: list[Agent], tool: str) -> list[str]:
    return [a.name for a in pool if tool in a.tools]
