"""Path template <-> regex helpers for discovering what's actually on disk.

We never hardcode the model/scenario coverage matrix -- `build_job_list.py`
globs the configured `adjusted_glob` template for each source family and
parses out whichever (model, scenario) combinations actually exist. This
way the pipeline picks up new models/scenarios/resolutions automatically
once `config.yaml`'s paths are updated, without any code changes.
"""

from __future__ import annotations

import glob
import re


def glob_and_regex(template: str) -> tuple[str, re.Pattern]:
    """Build a glob pattern and a matching regex from a `{model}`/{scenario}` template."""
    parts = re.split(r"(\{model\}|\{scenario\})", template)
    glob_chunks = []
    regex_chunks = []
    for part in parts:
        if part == "{model}":
            glob_chunks.append("*")
            regex_chunks.append(r"(?P<model>[^/]+)")
        elif part == "{scenario}":
            glob_chunks.append("*")
            regex_chunks.append(r"(?P<scenario>[^/]+)")
        else:
            glob_chunks.append(part)
            regex_chunks.append(re.escape(part))
    glob_pattern = "".join(glob_chunks)
    regex = re.compile("".join(regex_chunks) + "$")
    return glob_pattern, regex


def discover_model_scenario_paths(template: str) -> list[dict]:
    """Return [{"model": ..., "scenario": ..., "path": ...}, ...] for files matching template."""
    glob_pattern, regex = glob_and_regex(template)
    results = []
    for path in sorted(glob.glob(glob_pattern)):
        m = regex.match(path)
        if not m:
            continue
        results.append({"model": m["model"], "scenario": m["scenario"], "path": path})
    return results
