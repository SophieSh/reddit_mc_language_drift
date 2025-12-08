from __future__ import annotations
import yaml
import logging
import subprocess
import pathlib
import os

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

