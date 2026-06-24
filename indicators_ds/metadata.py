"""Attribute generation for the master indicators_ds output.

This module contains zero hardcoded descriptive text. Every string that
ends up in an output attribute is sourced from config.yaml's `metadata:`
section (or other config.yaml values, like era years / ensemble members /
indicator definitions). Functions here only assemble computed values
(era line lists, ensemble member lists, period month names) into the
templates config.yaml provides via str.format(). This keeps config.yaml
the single place a user edits to change any descriptive text in the
output -- editing this file should only ever be necessary to change *what
gets computed*, never to change wording.
"""

from __future__ import annotations

import calendar
import datetime

from config import Config


def _dim(config: Config, name: str) -> dict:
    return config.metadata["dimensions"][name]


def era_attrs(config: Config) -> dict:
    d = _dim(config, "era")
    lines = " ".join(
        d["line_template"].format(name=e.name, start_year=e.start_year, end_year=e.end_year)
        for e in config.eras
    )
    return {
        "long_name": d["long_name"],
        "description": d["description_template"].format(era_lines=lines),
    }


def model_attrs(config: Config) -> dict:
    d = _dim(config, "model")
    return {
        "long_name": d["long_name"],
        "description": d["description_template"].format(
            ensemble_name=config.ensemble_name,
            n_members=len(config.ensemble_members),
            members=", ".join(config.ensemble_members),
            reference_model=config.reference_model,
        ),
    }


def period_attrs(config: Config) -> dict:
    d = _dim(config, "period")
    lines = []
    for p in config.periods:
        month_names = "-".join(calendar.month_abbr[m] for m in p.months)
        wrap_suffix = d.get("wrap_suffix", "") if p.wrap else ""
        lines.append(d["line_template"].format(name=p.name, month_names=month_names, wrap_suffix=wrap_suffix))
    return {
        "long_name": d["long_name"],
        "description": d["description_template"].format(period_lines="; ".join(lines)),
    }


def scenario_attrs(config: Config) -> dict:
    d = _dim(config, "scenario")
    return {"long_name": d["long_name"], "description": d["description_template"]}


def aggregation_attrs(config: Config) -> dict:
    d = _dim(config, "aggregation")
    return {"long_name": d["long_name"], "description": d["description_template"]}


def variable_attrs(output_var: str, config: Config) -> dict:
    indicator = next(i for i in config.indicators.values() if i.output_var == output_var)
    return {
        "long_name": indicator.long_name,
        "units": indicator.units,
        "description": f"{indicator.description} {config.metadata['aggregation_method_note']}",
    }


def global_attrs(config: Config) -> dict:
    g = config.metadata["global_attrs"]
    timestamp = datetime.datetime.utcnow().isoformat()
    return {
        "title": g["title"],
        "institution": g["institution"],
        "summary": g["summary"],
        "Conventions": g["conventions"],
        "history": g["history_template"].format(timestamp=timestamp),
    }
