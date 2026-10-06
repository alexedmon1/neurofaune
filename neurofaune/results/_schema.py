"""A minimal JSON Schema validator for the keywords the results schemas use.

The schemas in ``schemas/`` are plain JSON Schema (draft 2020-12) and work with any
validator; this one exists so the checker needs only the standard library. It
supports exactly the keywords below and refuses a schema that uses any other, so a
schema cannot quietly rely on a keyword this validator would ignore.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).parent / "schemas"

_ANNOTATIONS = {"$schema", "$id", "title", "description"}
_KEYWORDS = {"type", "const", "enum", "required", "properties", "additionalProperties", "items",
             "minItems", "minLength", "minimum", "minProperties"}
_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """``name`` is analysis, provenance or columns."""
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def validate(value: Any, schema: dict, where: str = "$") -> list[str]:
    """Every way ``value`` departs from ``schema``, as ``"<path>: <problem>"`` strings."""
    unknown = set(schema) - _KEYWORDS - _ANNOTATIONS
    if unknown:
        raise ValueError(f"schema at {where} uses unsupported keywords {sorted(unknown)}")
    errors: list[str] = []
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[k](value) for k in kinds):
            return [f"{where}: expected {' or '.join(kinds)}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: must be {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{where}: must not be empty")
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and "minimum" in schema and value < schema["minimum"]):
        errors.append(f"{where}: {value} is below the minimum {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{where}: needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], f"{where}[{i}]")
    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{where}: needs at least {schema['minProperties']} entr(ies)")
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{where}: missing required field {key!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], f"{where}.{key}")
            elif extra is False:
                errors.append(f"{where}: unexpected field {key!r}")
            elif isinstance(extra, dict):
                errors += validate(item, extra, f"{where}.{key}")
    return errors
