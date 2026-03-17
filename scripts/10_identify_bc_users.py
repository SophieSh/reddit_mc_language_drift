#!/usr/bin/env python3
"""Step 10: Identify users who use/started/stopped birth control pills.

Two-stage pipeline:
  1. Regex: label-specific patterns for STARTED / STOPPED / STABLE_USING.
     Each pattern is precise enough to use without LLM (--no-llm flag).
  2. LLM (local Llama via ollama): re-classify or verify ambiguous cases.

Labels:
  STARTED      - user started BC pills (recently or at a stated point in time)
  STOPPED      - user stopped/quit BC pills (recently or at a stated point in time)
  STABLE_USING - user is stably on BC pills with no change event mentioned
  OTHER        - mentions BC pills but none of the above (someone else, hypothetical,
                 considering, afraid, general question, etc.)

Output columns designed for before/after analysis:
  author, offset_from_cd1, matched_term, regex_label, [llm_label, llm_timing, llm_reason]

Input:
  data/interim/timeline_with_offsets_with_anchors_*.csv

Output:
  data/interim/bc_candidates_{timestamp}.csv      -- regex hits with regex_label
  data/interim/bc_users_{timestamp}.csv           -- final results (regex or LLM)
  data/interim/bc_users_event_{timestamp}.csv     -- STARTED + STOPPED only
  data/interim/bc_users_stable_{timestamp}.csv    -- STABLE_USING only
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
# BC pill regex — label-specific patterns
# ---------------------------------------------------------------------------
# Strategy: each pattern group is specific enough to auto-label without LLM.
# _PILL matches any generic or brand pill term (used inside action patterns).
# Exclusion patterns are checked first to drop hypotheticals / other-person posts.
# ---------------------------------------------------------------------------

_BRAND_NAMES = (
    r"Yaz|Yasmin|Yasminelle|Junel|Sprintec|Tri.Sprintec|Loestrin|Lo\.?\s*Loestrin|"
    r"Microgestin|Ortho.Tri.Cyclen|Levlen|Alesse|Aviane|Lutera|Portia|Cryselle|"
    r"Blisovi|Seasonique|Seasonale|Lybrel|Mircette|Kariva|Estrostep|Nordette|"
    r"Levora|TriNessa|Elinest|Camrese|Introvale|Quasense|Daysee|Amethia|Chateal|"
    r"Falmina|Larissia|Ocella|Zarah|Novynette|Dianette|Cilest|Mercilon|Marvelon|"
    r"Microgynon|Rigevidon|Levest|Gedarel|Femodene|Femodette|Millinette|Logynon|"
    r"Trinovum|Brevicon|Modicon|Nelova|Nortrel|Ortho.Novum|Zenchent|Gianvi|"
    r"Loryna|Vestura|Nikki|Sylara|Cyred|Eminique|Aurovela|Larin|Natazia|Qlaira|"
    r"Zoely|Slinda|Slynd|Camila|Errin|Jencycla|Lyza|Nora.BE|"
    r"Norethindrone|Norgestrel|Desogestrel|Drospirenone"
)

# Any pill reference (generic or brand)
_PILL = (
    r"(?:the pill|bc pills?|birth control pills?|oral contraceptives?|"
    r"combined pill|mini.?pill|minipill|progestin.only pill|"
    r"BCP|BCPs|OCP|OCPs|contraceptive pill|"
    + _BRAND_NAMES + r")"
)

# Temporal / recency markers
_RECENT = r"(?:just|recently|finally|today|yesterday|last (?:week|month|night)|this (?:week|month)|a (?:few |couple of )?(?:days?|weeks?) ago|\d+ (?:days?|weeks?|months?) ago)"

# --- EXCLUSION: filter these out before labeling ----------------------------
_EXCLUSION_PATTERNS = [
    # Hypothetical / considering
    r"(?:thinking about|considering|want to|wondering if|should i|looking into|"
    r"debating|not sure (?:if|about)|deciding whether|contemplating|afraid to|"
    r"nervous about|scared to)\s+.{0,60}(?:start|take|go on|try)\s+.{0,40}" + _PILL,
    # Another person's pills (not the author)
    r"(?:my|her|his|their)\s+(?:friend|sister|mom|mother|partner|husband|boyfriend|"
    r"wife|daughter|girlfriend|roommate)\s+.{0,40}(?:started|takes|is on|taking|stopped|quit|prescribed)\s+.{0,30}" + _PILL,
    # Emergency / morning-after only
    r"\b(?:plan b|morning.?after pill|emergency contracepti)",
]

_EXCLUSION_REGEX = re.compile(
    "|".join(_EXCLUSION_PATTERNS), flags=re.IGNORECASE
)

# --- STARTED patterns -------------------------------------------------------
_STARTED_PATTERNS = [
    # "just/recently started (taking) <pill>"
    rf"{_RECENT}\s+started\s+(?:taking\s+)?{_PILL}",
    rf"started\s+(?:taking\s+)?{_PILL}\s+{_RECENT}",
    # "began taking <pill>"
    rf"(?:just |recently )?began\s+(?:taking\s+)?{_PILL}",
    # "put/started me on <pill>"
    rf"(?:put|started|got)\s+me\s+on\s+{_PILL}",
    rf"(?:my\s+)?(?:doctor|ob|gyn|gynecologist|physician)\s+.{{0,30}}(?:prescribed|put me on|started me on)\s+{_PILL}",
    # "prescribed <pill> and I start / started"
    rf"prescribed\s+{_PILL}.{{0,60}}(?:start|starting|started)",
    # "I'm on my first/second week/pack of <pill>"
    rf"(?:i'?m?|i am)\s+on\s+(?:my\s+)?(?:first|second|third|1st|2nd|3rd)\s+(?:day|week|pack|month)\s+(?:of|on)\s+{_PILL}",
    # "first week/pack on <pill>"
    rf"first\s+(?:week|pack|month)\s+(?:of|on)\s+{_PILL}",
    # "I started <brand> [timeframe]"
    rf"i\s+(?:just\s+)?started\s+(?:taking\s+)?(?:{_BRAND_NAMES})\b",
    # "going to start / about to start <pill>"
    rf"(?:going to|about to|starting)\s+(?:take\s+|start\s+)?{_PILL}\s+(?:tomorrow|next week|soon|this week)",
    # "switched to <pill> [recently]"
    rf"switched\s+(?:to|onto)\s+{_PILL}",
    # "<pill> for [X months] now" implying recent start
    rf"{_PILL}\s+for\s+(?:about\s+)?(?:a\s+)?(?:few\s+)?(?:1|2|3|4|5|6|one|two|three|four|five|six)\s+(?:days?|weeks?|months?)\s+(?:now|so far)",
]

# --- STOPPED patterns -------------------------------------------------------
_STOPPED_PATTERNS = [
    # "came/went off <pill>"
    rf"{_RECENT}\s+(?:came|went|gotten|got)\s+off\s+{_PILL}",
    rf"(?:came|went|gotten|got)\s+off\s+{_PILL}\s+{_RECENT}",
    # "stopped/quit/ditched taking <pill>"
    rf"{_RECENT}\s+(?:stopped|quit|ditched|dropped)\s+(?:taking\s+)?{_PILL}",
    rf"(?:stopped|quit|ditched|dropped)\s+(?:taking\s+)?{_PILL}\s+{_RECENT}",
    # "been off <pill> for [time]"
    rf"(?:i'?ve?\s+)?been\s+off\s+{_PILL}\s+for\s+(?:about\s+)?(?:a\s+)?(?:\d+|few|several|couple|a\s+while)",
    # "after/since coming/going off <pill>"
    rf"(?:after|since)\s+(?:stopping|quitting|coming\s+off|going\s+off)\s+{_PILL}",
    # "coming off / going off <pill>"
    rf"(?:coming|going|getting)\s+off\s+{_PILL}",
    # "stopped <brand>"
    rf"(?:stopped|quit|came off)\s+(?:{_BRAND_NAMES})\b",
    # "off the pill for [time]"
    rf"off\s+the\s+pill\s+for\s+(?:\d+|a\s+(?:few|couple)|several)\s+(?:days?|weeks?|months?)",
    # "no longer on <pill>"
    rf"no\s+longer\s+(?:on|taking)\s+{_PILL}",
]

# --- STABLE_USING patterns --------------------------------------------------
_STABLE_PATTERNS = [
    # "been on <pill> for [long time]"
    rf"(?:i'?ve?\s+)?been\s+(?:on|taking)\s+{_PILL}\s+for\s+(?:\d+|a\s+few|several|many)\s+(?:months?|years?)",
    # "on <pill> for years/months"
    rf"(?:on|taking)\s+{_PILL}\s+for\s+(?:\d+|a\s+few|several|many)\s+(?:months?|years?)",
    # "I take <pill> every day / daily"
    rf"(?:i\s+take|i'?m\s+taking)\s+{_PILL}\s+(?:every\s+day|daily|each\s+day)",
    # "currently on / still on <pill>"
    rf"(?:currently|still)\s+(?:on|taking)\s+{_PILL}",
    # "I'm on <pill> and ..." (stable context)
    rf"i'?m\s+on\s+{_PILL}\s+(?:and|for|since|because|to\s+(?:help|treat|manage|control))",
    # "have been on <pill> since"
    rf"(?:have|had)\s+been\s+(?:on|taking)\s+{_PILL}\s+since",
    # "<brand> for [long time]"
    rf"(?:{_BRAND_NAMES})\s+for\s+(?:\d+|a\s+few|several|many)\s+(?:months?|years?)",
    # "taking <brand> [daily/every day]"
    rf"taking\s+(?:{_BRAND_NAMES})\b.{{0,30}}(?:every\s+day|daily|for\s+(?:\d+|a\s+few|several)\s+(?:months?|years?))",
]

_STARTED_REGEX  = re.compile("|".join(_STARTED_PATTERNS),  flags=re.IGNORECASE)
_STOPPED_REGEX  = re.compile("|".join(_STOPPED_PATTERNS),  flags=re.IGNORECASE)
_STABLE_REGEX   = re.compile("|".join(_STABLE_PATTERNS),   flags=re.IGNORECASE)

# Combined filter: any post matching at least one label pattern is a candidate
_BC_REGEX = re.compile(
    "|".join(_STARTED_PATTERNS + _STOPPED_PATTERNS + _STABLE_PATTERNS),
    flags=re.IGNORECASE,
)


def classify_by_regex(text: str) -> str:
    """Return STARTED / STOPPED / STABLE_USING / EXCLUDED based on regex only."""
    t = str(text)
    if _EXCLUSION_REGEX.search(t):
        return "EXCLUDED"
    if _STARTED_REGEX.search(t):
        return "STARTED"
    if _STOPPED_REGEX.search(t):
        return "STOPPED"
    if _STABLE_REGEX.search(t):
        return "STABLE_USING"
    return "OTHER"  # matched combined but no specific label (shouldn't happen often)


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
         candidates_file: str | None = None, limit: int | None = None,
         no_llm: bool = False):
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

        # Apply regex labeling to all candidates
        print("  Applying label-specific regex classification...")
        candidates_slim["regex_label"] = candidates_slim["text"].apply(classify_by_regex)

        # Print regex label distribution
        print("\n  Regex label distribution:")
        for label, count in candidates_slim["regex_label"].value_counts().items():
            pct = 100 * count / len(candidates_slim)
            marker = " ✓" if label in EVENT_LABELS | STABLE_LABELS else ""
            print(f"    {label:<18} {count:>5}  ({pct:.1f}%){marker}")

        out_candidates = save_with_timestamp(candidates_slim, interim_dir, "bc_candidates")
        print(f"\n  Saved → {out_candidates.name}")

    if limit:
        print(f"\n  Limiting to first {limit} rows (--limit {limit})")
        candidates_slim = candidates_slim.head(limit)

    # ---- LLM classification (optional) ---------------------------------------
    if no_llm:
        print("\n[Stage 3] Skipping LLM — using regex labels only.")
        classified = candidates_slim.copy()
        # Drop EXCLUDED and OTHER — keep only clearly labeled posts
        classified = classified[classified["regex_label"].isin(EVENT_LABELS | STABLE_LABELS)].copy()
        print(f"  Kept {len(classified):,} posts with clear regex labels "
              f"({classified['author'].nunique():,} users)")
        label_col = "regex_label"
    else:
        print(f"\n[Stage 3] LLM classification with '{model}'...")
        classified = classify_batch(candidates_slim, model=model)
        label_col = "llm_label"

    # Add text_snippet for easy reading
    classified["text_snippet"] = classified["text"].str[:300]

    out_all = save_with_timestamp(classified, interim_dir, "bc_users")
    print(f"\n  Saved all results → {out_all.name}")

    # Label distribution
    print(f"\n  Label distribution ({label_col}):")
    for label, count in classified[label_col].value_counts().items():
        pct = 100 * count / len(classified)
        marker = " ✓" if label in EVENT_LABELS | STABLE_LABELS else ""
        print(f"    {label:<18} {count:>5}  ({pct:.1f}%){marker}")

    # ---- Split into research-ready subsets -----------------------------------
    event_df = classified[classified[label_col].isin(EVENT_LABELS)].copy()
    print(f"\n  Event group (STARTED+STOPPED): "
          f"{event_df['author'].nunique():,} users, {len(event_df):,} posts")
    if not event_df.empty:
        out_event = save_with_timestamp(event_df, interim_dir, "bc_users_event")
        print(f"  Saved → {out_event.name}")

    stable_df = classified[classified[label_col].isin(STABLE_LABELS)].copy()
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
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM stage — use label-specific regex labels only.",
    )
    args = ap.parse_args()
    exit(main(config_path=args.config, model=args.model,
              candidates_file=args.candidates, limit=args.limit,
              no_llm=args.no_llm))
