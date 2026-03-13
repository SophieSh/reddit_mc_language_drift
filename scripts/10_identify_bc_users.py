#!/usr/bin/env python3
"""Step 10: Identify users who use/started/stopped birth control pills.

Two-stage pipeline:
  1. Regex: broadly match any mention of BC pills or pill brand names.
  2. LLM (local Llama via ollama): classify the user's relationship to BC pills
     AND extract timing information (when they started/stopped).

Labels:
  STARTED      - user started BC pills (recently or at a stated point in time)
  STOPPED      - user stopped/quit BC pills (recently or at a stated point in time)
  STABLE_USING - user is stably on BC pills with no change event mentioned
  OTHER        - mentions BC pills but none of the above (someone else, hypothetical,
                 considering, afraid, general question, etc.)

Output columns designed for before/after analysis:
  author, offset_from_cd1, matched_term, llm_label, llm_timing, llm_reason, text_snippet

Input:
  data/interim/timeline_with_offsets_with_anchors_*.csv

Output:
  data/interim/bc_candidates_{timestamp}.csv      -- regex hits (before LLM)
  data/interim/bc_users_{timestamp}.csv           -- all LLM results
  data/interim/bc_users_event_{timestamp}.csv     -- STARTED + STOPPED only (event-based)
  data/interim/bc_users_stable_{timestamp}.csv    -- STABLE_USING only (baseline group)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc)
        return iterable

from src.config import load_config
from src.io import find_latest_file, save_with_timestamp


# ---------------------------------------------------------------------------
# BC pill regex patterns (broad — all intent/timing handled by LLM)
# ---------------------------------------------------------------------------

_GENERIC = [
    r"birth[ -]control pill",
    r"oral contraceptive",
    r"\bthe pill\b",
    r"\bBCP\b",
    r"\bBCPs\b",
    r"\bOCP\b",
    r"\bOCPs\b",
    r"\bbc pill",
    r"\btaking bc\b",
    r"\bstarted bc\b",
    r"\bstopped bc\b",
    r"\bquit bc\b",
    r"\bcontraceptive pill",
    r"\bminipill\b",
    r"\bmini.pill\b",
    r"\bprogestin.only pill\b",
    r"\bcombined pill\b",
    r"been on the pill",
    r"started.{0,10}pill",
    r"taking.{0,10}pill",
    r"stopped.{0,10}pill",
    r"quit.{0,10}pill",
    r"went on.{0,10}pill",
    r"went off.{0,10}pill",
    r"coming off.{0,10}pill",
    r"came off.{0,10}pill",
    r"got on.{0,10}pill",
    r"got off.{0,10}pill",
]

# Brand / generic drug names (oral contraceptive pills only)
_BRANDS = [
    r"\bYaz\b",
    r"\bYasmin\b",
    r"\bYasminelle\b",
    r"\bJunel\b",
    r"\bSprintec\b",
    r"\bLoestrin\b",
    r"\bLo Loestrin\b",
    r"\bMicrogestin\b",
    r"\bOrtho.Tri.Cyclen\b",
    r"\bTri.Sprintec\b",
    r"\bLevlen\b",
    r"\bAlesse\b",
    r"\bAviane\b",
    r"\bLutera\b",
    r"\bPortia\b",
    r"\bCryselle\b",
    r"\bBlisovi\b",
    r"\bSeasonique\b",
    r"\bSeasonale\b",
    r"\bLybrel\b",
    r"\bMircette\b",
    r"\bKariva\b",
    r"\bEstrostep\b",
    r"\bNordette\b",
    r"\bLevora\b",
    r"\bTriNessa\b",
    r"\bElinest\b",
    r"\bCamrese\b",
    r"\bIntrovale\b",
    r"\bQuasense\b",
    r"\bDaysee\b",
    r"\bAmethia\b",
    r"\bChateal\b",
    r"\bFalmina\b",
    r"\bLarissia\b",
    r"\bOcella\b",
    r"\bZarah\b",
    r"\bNovynette\b",
    r"\bDianette\b",
    r"\bCilest\b",
    r"\bMercilon\b",
    r"\bMarvelon\b",
    r"\bMicrogynon\b",
    r"\bRigevidon\b",
    r"\bLevest\b",
    r"\bGedarel\b",
    r"\bFemodene\b",
    r"\bFemodette\b",
    r"\bMillinette\b",
    r"\bLogynon\b",
    r"\bTrinovum\b",
    r"\bBrevicon\b",
    r"\bModicon\b",
    r"\bNelova\b",
    r"\bNortrel\b",
    r"\bOrtho.Novum\b",
    r"\bZenchent\b",
    r"\bGianvi\b",
    r"\bLoryna\b",
    r"\bVestura\b",
    r"\bNikki\b",
    r"\bSylara\b",
    r"\bCyred\b",
    r"\bEminique\b",
    r"\bAurovela\b",
    r"\bLarin\b",
    r"\bNatazia\b",
    r"\bQlaira\b",
    r"\bZoely\b",
    r"\bSlinda\b",
    r"\bSlynd\b",
    r"\bCamila\b",
    r"\bErrin\b",
    r"\bJencycla\b",
    r"\bLyza\b",
    r"\bNora.BE\b",
    r"\bNorethindrone\b",
    r"\bNorgestrel\b",
    r"\bDesogestrel\b",
    r"\bDrospirenone\b",
]

_BC_REGEX = re.compile(
    "|".join(_GENERIC + _BRANDS),
    flags=re.IGNORECASE,
)


def find_matched_term(text: str) -> str:
    m = _BC_REGEX.search(str(text))
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# LLM classification via ollama
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

LABELS = ["STARTED", "STOPPED", "STABLE_USING", "OTHER"]

SYSTEM_PROMPT = """You are a medical text classification assistant analyzing Reddit posts about birth control pills.

Classify the post author's relationship to birth control pills using EXACTLY one of these labels:

STARTED
  - The author recently started taking birth control pills, OR is starting within the next few days.
  - Includes: "I just started the pill", "started Yaz last week", "my doctor prescribed me bc pills and I start tomorrow", "been on it for 2 months now" (if this is a new thing for them).
  - Key signal: a clear start event with approximate timing.

STOPPED
  - The author recently stopped or quit birth control pills, OR is stopping imminently.
  - Includes: "I quit the pill last month", "just came off Yasmin", "stopped taking bc pills 3 weeks ago", "going off the pill tomorrow".
  - Key signal: a clear stop event with approximate timing.

STABLE_USING
  - The author is currently on birth control pills as a stable ongoing condition, with NO start or stop event described.
  - Includes: "I take Sprintec every day", "been on the pill for years", "I'm on bc pills and my periods are regular", "taking Yaz for PCOS".
  - Key signal: pill use is background context, not a change event.

OTHER
  - Anything that does not fit STARTED, STOPPED, or STABLE_USING.
  - Use this for: someone else's pill use, hypothetical/considering/afraid, general questions about pills, mentions of pills in past history only (years ago, no current relevance), morning-after pill only (not regular BC pills).

IMPORTANT RULES:
- If the author says "I am afraid to start" or "thinking about starting" → OTHER (not STARTED).
- If the author says "my friend started the pill" → OTHER (not STARTED).
- If the author says "I started it a few years ago" with no current relevance → OTHER.
- A post about someone getting PRESCRIBED pills but not yet started → STARTED only if they say they will start imminently (tomorrow, this week). Otherwise → OTHER.

Also extract the TIMING: quote or closely paraphrase what the post says about when the start/stop happened or how long they have been using. If no timing info, write "not specified".

Reply ONLY with a JSON object with exactly three fields: "label", "timing", "reason".
- label: one of STARTED, STOPPED, STABLE_USING, OTHER
- timing: verbatim or close paraphrase of timing from the post, or "not specified"
- reason: one sentence explaining your classification

Example outputs:
{"label": "STARTED", "timing": "started last week", "reason": "Author says they just started Yaz last week for acne."}
{"label": "STOPPED", "timing": "stopped 3 weeks ago", "reason": "Author says they came off the pill 3 weeks ago and their cycle is irregular."}
{"label": "STABLE_USING", "timing": "been on it for 2 years", "reason": "Author describes being on Sprintec for 2 years as a stable condition."}
{"label": "OTHER", "timing": "not specified", "reason": "Author is asking about whether to start the pill, not currently taking it."}
"""


def classify_post(text: str, model: str, timeout: int = 60) -> dict:
    """Send a single post to ollama and return parsed label + timing + reason."""
    text_truncated = str(text)[:2000]

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": f"Classify this Reddit post:\n\n{text_truncated}",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Strip markdown code fences if present
        json_str = raw
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    json_str = part
                    break

        parsed = json.loads(json_str)
        label = parsed.get("label", "OTHER").strip().upper()
        if label not in LABELS:
            label = "OTHER"
        return {
            "label": label,
            "timing": parsed.get("timing", "not specified"),
            "reason": parsed.get("reason", ""),
            "raw_response": raw,
        }

    except requests.exceptions.ConnectionError:
        return {
            "label": "ERROR",
            "timing": "",
            "reason": "ollama not running — start with: ollama serve",
            "raw_response": "",
        }
    except Exception as e:
        return {"label": "ERROR", "timing": "", "reason": str(e), "raw_response": ""}


def classify_batch(
    df: pd.DataFrame,
    model: str,
    text_col: str = "text",
) -> pd.DataFrame:
    """Classify all rows. Adds columns: llm_label, llm_timing, llm_reason, llm_raw."""
    labels, timings, reasons, raws = [], [], [], []

    n = len(df)
    print(f"  Classifying {n:,} candidate posts with model '{model}'...")

    label_counts: dict[str, int] = {}

    with tqdm(total=n, unit="post", dynamic_ncols=True) as bar:
        for i, text in enumerate(df[text_col], start=1):
            result = classify_post(text, model)
            labels.append(result["label"])
            timings.append(result["timing"])
            reasons.append(result["reason"])
            raws.append(result["raw_response"])

            # Track label counts for postfix display
            lbl = result["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

            bar.set_postfix({
                "last": lbl,
                "S": label_counts.get("STARTED", 0),
                "X": label_counts.get("STOPPED", 0),
                "U": label_counts.get("STABLE_USING", 0),
                "O": label_counts.get("OTHER", 0),
                "ERR": label_counts.get("ERROR", 0),
            }, refresh=False)
            bar.update(1)

            if result["label"] == "ERROR":
                tqdm.write(f"  ERROR on row {i}: {result['reason']}")
                if i == 1:
                    tqdm.write("  Aborting — check ollama is running (`ollama serve`)")
                    tqdm.write(f"  and model is pulled (`ollama pull {model}`)")
                    break

    df = df.copy()
    pad = n - len(labels)
    df["llm_label"]  = labels  + [None] * pad
    df["llm_timing"] = timings + [None] * pad
    df["llm_reason"] = reasons + [None] * pad
    df["llm_raw"]    = raws    + [None] * pad
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EVENT_LABELS  = {"STARTED", "STOPPED"}
STABLE_LABELS = {"STABLE_USING"}


def main(config_path: str = "configs/base.yaml", model: str = "llama3.1:8b",
         candidates_file: str | None = None, limit: int | None = None):
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 10: Identify Birth Control Pill Users")
    print("=" * 60)
    print()

    if candidates_file:
        # ---- Load existing candidates checkpoint -----------------------------
        cpath = Path(candidates_file)
        print(f"[Stage 2] Loading existing candidates checkpoint: {cpath.name}")
        candidates_slim = pd.read_csv(cpath, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(candidates_slim):,} candidates from checkpoint")
    else:
        # ---- Load timeline ---------------------------------------------------
        print("[Stage 1] Loading timeline with offsets...")
        timeline_file = find_latest_file(interim_dir, "timeline_with_offsets_with_anchors_*.csv")
        if not timeline_file:
            raise FileNotFoundError(
                f"No timeline file found in {interim_dir}.\n"
                "Run scripts/05_build_timeline.py first."
            )
        print(f"  File: {timeline_file.name}")
        df = pd.read_csv(timeline_file, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(df):,} posts from {df['author'].nunique():,} users")

        for col in ("author", "offset_from_cd1", "text"):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing. Available: {df.columns.tolist()}")

        # ---- Regex filter ----------------------------------------------------
        print("\n[Stage 2] Applying BC pill regex filter...")
        mask = df["text"].apply(lambda t: bool(_BC_REGEX.search(str(t))))
        candidates = df[mask].copy()
        candidates["matched_term"] = candidates["text"].apply(find_matched_term)
        print(f"  Regex hits: {len(candidates):,} posts from {candidates['author'].nunique():,} users"
              f"  ({100 * len(candidates) / len(df):.1f}% of timeline)")

        if candidates.empty:
            print("  No candidates found.")
            return 1

        # Slim to needed columns only
        keep_cols = ["author", "offset_from_cd1", "matched_term", "text"]
        for ts_col in ("ts_utc", "created_utc", "timestamp"):
            if ts_col in candidates.columns:
                keep_cols.append(ts_col)
                break
        candidates_slim = candidates[keep_cols].copy()

        out_candidates = save_with_timestamp(candidates_slim, interim_dir, "bc_candidates")
        print(f"  Saved → {out_candidates.name}")

    if limit:
        print(f"\n  Limiting to first {limit} rows (--limit {limit})")
        candidates_slim = candidates_slim.head(limit)

    # ---- LLM classification --------------------------------------------------
    print(f"\n[Stage 3] LLM classification with '{model}'...")
    classified = classify_batch(candidates_slim, model=model)

    # Add text_snippet column for easy reading (300 chars)
    classified["text_snippet"] = classified["text"].str[:300]

    # Full results (drop raw LLM response to keep file readable, keep for debugging)
    out_all = save_with_timestamp(classified, interim_dir, "bc_users")
    print(f"\n  Saved all results → {out_all.name}")

    # Label distribution
    print("\n  Label distribution:")
    for label, count in classified["llm_label"].value_counts().items():
        pct = 100 * count / len(classified)
        marker = " ✓" if label in EVENT_LABELS | STABLE_LABELS else ""
        print(f"    {label:<18} {count:>5}  ({pct:.1f}%){marker}")

    # ---- Split into research-ready subsets -----------------------------------
    # Event group: STARTED + STOPPED — for before/after analysis
    event_df = classified[classified["llm_label"].isin(EVENT_LABELS)].copy()
    print(f"\n  Event group (STARTED+STOPPED): "
          f"{event_df['author'].nunique():,} users, {len(event_df):,} posts")
    if not event_df.empty:
        out_event = save_with_timestamp(event_df, interim_dir, "bc_users_event")
        print(f"  Saved → {out_event.name}")

    # Stable group: STABLE_USING — for comparison baseline
    stable_df = classified[classified["llm_label"].isin(STABLE_LABELS)].copy()
    print(f"\n  Stable group (STABLE_USING): "
          f"{stable_df['author'].nunique():,} users, {len(stable_df):,} posts")
    if not stable_df.empty:
        out_stable = save_with_timestamp(stable_df, interim_dir, "bc_users_stable")
        print(f"  Saved → {out_stable.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Identify BC pill users from timeline posts")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model (default: llama3.1:8b). "
             "Alternatives: llama3.2:3b (faster), llama3.1:70b (better quality)",
    )
    ap.add_argument(
        "--candidates",
        default=None,
        help="Path to existing bc_candidates_*.csv to skip regex stage (checkpoint resume).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only classify the first N rows (for testing/sampling).",
    )
    args = ap.parse_args()
    exit(main(config_path=args.config, model=args.model,
              candidates_file=args.candidates, limit=args.limit))
