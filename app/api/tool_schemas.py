"""OpenAI-compatible function schemas for available system tools."""

from __future__ import annotations

from typing import Dict


TOOL_SCHEMAS: Dict[str, Dict] = {
    "read_sensors": {
        "name": "read_sensors",
        "description": "Read the most recent environmental sensor values.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "set_fan_speed": {
        "name": "set_fan_speed",
        "description": "Adjust the active cooling fan speed as a float between 0 and 1.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
    "read_state": {
        "name": "read_state",
        "description": "Return the persisted shared state snapshot.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "update_state": {
        "name": "update_state",
        "description": "Merge provided fields into the shared state store.",
        "parameters": {
            "type": "object",
            "properties": {
                "delta": {"type": "object"}
            },
            "required": ["delta"],
            "additionalProperties": False,
        },
    },
    "append_memory": {
        "name": "append_memory",
        "description": "Persist a memory string with optional metadata tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    "query_memory": {
        "name": "query_memory",
        "description": "Retrieve recent memories matching a keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
    "read_gpu_temps": {
        "name": "read_gpu_temps",
        "description": "Reads GPU temperatures using nvidia-smi.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


__all__ = ["TOOL_SCHEMAS"]
