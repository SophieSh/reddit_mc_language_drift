#!/usr/bin/env python3
"""Each eligible user's age at their anchor post, from self-disclosures in history.

For each eligible user: scan all posts (moon1+moon2) for a first-person age
disclosure, keep the LATEST one, then age_at_anchor = anchor_year - birth_year.
"""
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ELIG_FILE   = ROOT / "data/submission/processed/users_database_CD_20260329T064620.csv"
ANCHOR_FILE = ROOT / "data/submission/processed/users_database_CD_20260329T064620.csv"
POST_FILES  = [ROOT / "data/raw/moon1_all_post_2015_2025_filtered_features.csv",
               ROOT / "data/raw/moon2_all_post_2015_2025_filtered_features.csv"]
AGE_LO, AGE_HI = 13, 60

# block age numbers that are really a unit (weeks pregnant, lbs, ft, dpo, ...);
# the optional "or 13 / -13 / to 13" skip also catches ranges like "12 or 13 DPO"
_UNIT = (r"(?!\s*(?:(?:or|to|[-–])\s*\d{1,2}\s*)?(?:weeks?|wks?|months?|mos?|days?|hours?|hrs?|min(?:ute)?s?|dpo|lbs?|pounds?"
         r"|kgs?|kilos?|cm|mm|ft|feet|foot|inch(?:es)?|miles?|km|%|dollars?|usd|euros?|stone|grams?|times?|x\b))")

# Family 1 - verbal, first-person:  "I am 25", "I'm 25yo", "I turned 25", "me, 25 years old"
VERBAL = re.compile(
    r"\b(?:i\s*am|i['’]?m|im)\s+(?:(?:now|currently|only|just|barely|almost|nearly|about|a|an|finally)\s+)?"
    r"(\d{1,2})(?!\d)" + _UNIT + r"(?:\s*(?:years?\s*old|yrs?\s*old|y\s*/?\s*o))?"
    r"|\b(?:i\s*am|i['’]?m|im|i)\s*(?:just\s+|recently\s+|finally\s+)?turn(?:ed|ing)\s+(\d{1,2})(?!\d)"
    r"|\b(?:i|me)\s*\(\s*(\d{1,2})\s*\)"
    r"|\b(?:i|me)\b[\s,]+(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|y\s*/?\s*o)\b",
    re.IGNORECASE)

# Family 2 - sex-tag, first-person:  "Me [25F]", "I'm 25F", "I (F25)", "me 25f"
SEXTAG = re.compile(
    r"\b(?:i\s*am|i['’]?m|im|me|myself|i)\b[\s,:\-]*"
    r"(?:[\(\[]\s*(?:f\s*(\d{1,2})|(\d{1,2})\s*f)\s*[\)\]]|f\s*(\d{2})\b|(\d{2})\s*f\b)",
    re.IGNORECASE)


def coalesce(text, pat):                       # first non-null capture group -> age
    groups = text.str.extract(pat)
    return pd.to_numeric(groups.bfill(axis=1).iloc[:, 0], errors="coerce")


eligible = set(pd.read_csv(ELIG_FILE, usecols=["user"], encoding="utf-8-sig")["user"])

anchor_df = pd.read_csv(ANCHOR_FILE, usecols=["user", "timestep"], encoding="utf-8-sig")
anchor = dict(zip(anchor_df["user"], anchor_df["timestep"].str[:4].astype(int)))

latest = {}                                    # user -> (created_utc, age, kind, match, snippet, text)
for post_file in POST_FILES:
    for chunk in pd.read_csv(post_file, usecols=["author", "created_utc", "selftext", "title"],
                             encoding="utf-8-sig", chunksize=250_000):
        chunk = chunk[chunk["author"].isin(eligible)]
        text = (chunk["title"].fillna("") + " " + chunk["selftext"].fillna("")).str.replace(r"\s+", " ", regex=True)
        sextag, verbal = coalesce(text, SEXTAG), coalesce(text, VERBAL)
        age = sextag.fillna(verbal)            # sex-tag wins ties
        kind = sextag.notna().map({True: "sextag", False: "verbal"})
        valid = age.between(AGE_LO, AGE_HI)
        for author, created, age_val, mtype, post in zip(
                chunk["author"][valid], chunk["created_utc"][valid], age[valid].astype(int),
                kind[valid], text[valid]):
            if author not in latest or created > latest[author][0]:
                match = (SEXTAG if mtype == "sextag" else VERBAL).search(post)
                snippet = "…" + post[max(0, match.start() - 40):match.end() + 40] + "…"
                latest[author] = (int(created), age_val, mtype, match.group(0), snippet, post[:500])
    print(f"  {post_file.name}: {len(latest):,} users")

rows = []
for author, (created, age_val, mtype, match, snippet, post) in latest.items():
    post_year = pd.Timestamp(created, unit="s").year
    rows.append((author, mtype, age_val, match, post_year, post_year - age_val,
                 anchor.get(author), anchor.get(author, 0) - (post_year - age_val), snippet, post))
df = pd.DataFrame(rows, columns=["author", "match_type", "age_extracted", "matched_text",
                                 "post_year", "birth_year", "anchor_year", "age_at_anchor",
                                 "snippet", "text_excerpt"])
df = df[df["age_at_anchor"].between(AGE_LO, AGE_HI)].sort_values("age_at_anchor")

stamp = time.strftime("%Y%m%dT%H%M%S")
df.to_csv(ROOT / "data/processed" / f"age_disclosure_posts_{stamp}.csv", index=False)
df[["author", "match_type", "birth_year", "anchor_year", "age_at_anchor"]].to_csv(
    ROOT / "data/processed" / f"user_age_distribution_{stamp}.csv", index=False)
print(f"n {len(df):,}  median {df['age_at_anchor'].median():.0f}  mean {df['age_at_anchor'].mean():.1f}")

plt.figure(figsize=(7, 4))
plt.hist(df["age_at_anchor"], bins=range(AGE_LO, AGE_HI + 2), edgecolor="white")
plt.xlim(AGE_LO, 50)
plt.xlabel("Age at anchor post (year of post)")
plt.ylabel("Number of authors")
# no title: the caption is written under the figure in the paper
plt.tight_layout()
plt.savefig(ROOT / "data/processed" / f"user_age_distribution_{stamp}.png", dpi=130)
