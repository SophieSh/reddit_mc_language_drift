import re
import pandas as pd
from pathlib import Path
from datetime import datetime


# Constants
AVG_DAYS_PER_MONTH = 30.5


# /////////////////////////////////////////////////////////////////////////////
# BASIC DATA CLEANING
# /////////////////////////////////////////////////////////////////////////////

def filter_deleted_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where author is [deleted]."""
    return df[df['author'] != '[deleted]'].copy()


def concatenate_title_selftext_text(title: str | None, selftext: str | None) -> str:
    """Concatenate title and selftext into single text string.
    
    Args:
        title: Post title (can be None)
        selftext: Post selftext (can be None)
    
    Returns:
        Concatenated text string
    """
    title = title if title is not None else ""
    selftext = selftext if selftext is not None else ""
    return (title + " " + selftext).strip()


def concatenate_title_selftext(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate title and selftext into text column (vectorized for performance)."""
    df = df.copy()
    df['text'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.strip()
    return df


def filter_posts_by_anchor_window(
    posts_df: pd.DataFrame,
    users_df: pd.DataFrame,
    window_months: int = 12,
    user_col: str = "author",
    timestamp_col: str = "ts_utc"
) -> pd.DataFrame:
    """Filter posts to only those within ±N months of anchor timestamp.
    
    Reduces noise by keeping only posts within a time window around when
    the user explicitly mentioned their period (anchor post).
    
    Args:
        posts_df: DataFrame with posts (must have user_col and timestamp_col)
        users_df: DataFrame with user anchors (must have 'user' and 'timestep')
        window_months: Number of months before/after anchor to keep (default 12)
        user_col: Column name for user ID in posts_df
        timestamp_col: Column name for timestamp in posts_df (datetime-like)
    
    Returns:
        Filtered DataFrame with only posts within window
    """
    anchor_map = users_df.set_index('user')['timestep']
    
    posts_df = posts_df.copy()
    posts_df['_anchor_ts'] = posts_df[user_col].map(anchor_map)
    
    posts_df['_days_from_anchor'] = (
        pd.to_datetime(posts_df[timestamp_col]) - 
        pd.to_datetime(posts_df['_anchor_ts'])
    ).dt.days.abs()
    
    max_days = int(window_months * AVG_DAYS_PER_MONTH)
    filtered_df = posts_df[posts_df['_days_from_anchor'] <= max_days].copy()
    filtered_df = filtered_df.drop(columns=['_anchor_ts', '_days_from_anchor'])
    
    return filtered_df

def fix_removed_posts_selftext(df: pd.DataFrame) -> pd.DataFrame:
    """
    If selftext is '[removed]' or '[deleted]' (Reddit's markers for deleted posts), set selftext to ''.
    This ensures that, for posts where title is informative but selftext was removed, we don't carry over junk.
    """
    df = df.copy()
    #Creates a boolean Series (True/False mask)
    mask = df['selftext'].isin(['[removed]', '[deleted]'])
    df.loc[mask, 'selftext'] = ''
    return df


# /////////////////////////////////////////////////////////////////////////////
# PATTERN MATCHING AND EXTRACTION
# /////////////////////////////////////////////////////////////////////////////

def extract_sentence_with_match(text: str, pattern: str) -> str | None:
    """Extract the sentence containing the regex match.
    
    Handles Reddit-specific formatting:
    - Multiple punctuation (!!!, ???)
    - No space after punctuation
    - Multiple newlines
    """
    if pd.isna(text) or not text:
        return None
    
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    match_pos = match.start()
    
    sentence_boundary = r'[.!?]+\s*|\n+'
    
    start = 0
    for m in re.finditer(sentence_boundary, text[:match_pos]):
        start = m.end()
    
    end = len(text)
    for m in re.finditer(sentence_boundary, text[match_pos:]):
        end = match_pos + m.start() + 1
        break
    
    return text[start:end].strip()


def extract_matched_phrase(text: str, pattern: str) -> str | None:
    """Extract the exact phrase that matched the pattern."""
    if pd.isna(text) or not text:
        return None
    
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    return match.group(0)


def normalize_phrase(phrase: str | None) -> str | None:
    """Lowercase and collapse whitespace for grouping purposes."""
    if phrase is None or pd.isna(phrase):
        return None
    normalized = " ".join(str(phrase).strip().split()).lower()
    return normalized if normalized else None


# /////////////////////////////////////////////////////////////////////////////
# UNCERTAINTY DETECTION
# /////////////////////////////////////////////////////////////////////////////

def check_uncertainty_before_match(
    sentence: str,
    matched_phrase: str,
    window_words: int = 3,
) -> bool:
    """Check for hedge language immediately before the matched phrase.

    Looks at the final few words before the phrase, ignoring earlier sentences,
    and searches for uncertainty markers such as "I think", "maybe", etc.
    """
    if pd.isna(sentence) or pd.isna(matched_phrase):
        return False

    # Find the exact phrase location within the sentence
    phrase_pattern = re.compile(re.escape(matched_phrase), flags=re.IGNORECASE)
    phrase_match = phrase_pattern.search(sentence)
    if not phrase_match:
        return False

    prefix = sentence[: phrase_match.start()]

    # Limit to the most recent words before the phrase
    words = prefix.split()
    if not words:
        return False
    prefix = " ".join(words[-window_words:]).lower().strip()
    if not prefix:
        return False

    uncertainty_patterns = [
        r"\bi\s+(think|thought|guess|suspect|wonder|hope)\b",
        r"\bi\s+(might|may|could|probably|possibly|maybe)\b",
        r"\bmaybe\b",
        r"\bnot\s+sure\b",
        r"\bunsure\b",
        r"\bfeel(?:s)?\s+like\b",
        r"\bseems?\s+like\b",
        r"\bsort\s+of\b",
        r"\bkind\s+of\b",
        r"\balmost\b",
        r"\babout\s+to\b",
        r"\bshould(?:'ve| have)\b",
        r"\bsupposed\s+to\b",
        r"\bexpecting\b",
        r"\bexpected\b",
    ]

    for pattern in uncertainty_patterns:
        if re.search(pattern, prefix):
            return True

    # Question mark immediately before the phrase indicates uncertainty
    if prefix.endswith("?") or prefix.endswith("??"):
        return True

    return False


# /////////////////////////////////////////////////////////////////////////////
# CYCLE DAY OFFSET CALCULATION
# /////////////////////////////////////////////////////////////////////////////

# Type 1: Immediate start (pattern_2: "just got my period")
def calculate_offset_immediate_start() -> int:
    """Calculate offset for immediate period start: always 0."""
    return 0


# Type 2: Simple markers (yesterday, today, etc.)
SIMPLE_MARKERS = [
    ("a couple of days ago", 1),
    ("yesterday", 1),
    ("last night", 1),
    ("this morning", 0),
    ("this evening", 0),
    ("today", 0),
]


def calculate_offset_from_simple_marker(matched_phrase: str) -> int | None:
    """Calculate offset from simple markers like 'yesterday', 'today', etc.
    
    Examples: 'got my period today', 'started my period yesterday'
    """
    if pd.isna(matched_phrase):
        return None
    lowered = matched_phrase.lower()
    for marker, offset in SIMPLE_MARKERS:
        if marker in lowered:
            return offset
    return None


# Type 3: Explicit day number (pattern_5: "on day X of my period")
def calculate_offset_from_explicit_day(matched_phrase: str) -> int | None:
    """Extract day number from matched phrase and convert to offset.
    
    Pattern is "on day X of my period", so extract X directly.
    Day 1 = offset 0, Day 5 = offset 4, etc.
    """
    if pd.isna(matched_phrase):
        return None
    
    day_match = re.search(r"day (\d+)", matched_phrase.lower())
    if day_match:
        day_num = int(day_match.group(1))
        return day_num - 1
    
    return None


def extract_date_from_sentence(sentence: str, post_timestamp: pd.Timestamp | None = None) -> datetime | None:
    """Extract date from sentence like 'my period on January 17th', 'started on March 5', etc.
    
    Args:
        sentence: Text containing date
        post_timestamp: Post's publication timestamp for year inference
    
    Returns:
        Parsed datetime or None if no date found.
    """
    if pd.isna(sentence) or not sentence:
        return None
    
    months = {
        'january': 1,
        'february': 2,
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8,
        'september': 9,
        'october': 10,
        'november': 11,
        'december': 12,
    }
    
    pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{1,2})(?:st|nd|rd|th)?\b'
    
    match = re.search(pattern, sentence.lower())
    if not match:
        return None
    
    month_name = match.group(1)
    day = int(match.group(2))
    month = months.get(month_name)
    
    if not month or day < 1 or day > 31:
        return None
    
    if post_timestamp is None:
        year = 2000
    else:
        post_date = post_timestamp.to_pydatetime().replace(tzinfo=None)
        post_year = post_date.year
        post_month = post_date.month
        
        if month > post_month:
            year = post_year - 1
        else:
            year = post_year
    
    try:
        return datetime(year=year, month=month, day=day)
    except ValueError:
        return None


# Type 4: Date-based (pattern_3: "my period on [Month] [Day]")
def extract_date_after_phrase(sentence: str, phrase: str, window_words: int = 3, post_timestamp: pd.Timestamp | None = None) -> datetime | None:
    """Extract date immediately after "my period on" phrase.
    
    Structure is known: "my period on [Month] [Day]" - extract month and day directly.
    
    Args:
        sentence: Full sentence containing "my period on [Month] [Day]"
        phrase: Phrase to search for ("my period on")
        window_words: Number of words after phrase to search (default 3)
        post_timestamp: Post's publication timestamp for year inference
    
    Returns:
        Parsed datetime or None if no date found.
    """
    if pd.isna(sentence) or not sentence or pd.isna(phrase):
        return None
    
    lowered_sentence = sentence.lower()
    lowered_phrase = phrase.lower()
    
    phrase_pos = lowered_sentence.find(lowered_phrase)
    if phrase_pos == -1:
        return None
    
    after_phrase = sentence[phrase_pos + len(phrase):]
    words = after_phrase.split()[:window_words]
    
    if len(words) < 1:
        return None
    
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }
    
    month_name = None
    day_str = None
    
    for i, word in enumerate(words):
        word_lower = word.lower().rstrip('.,!?;:')
        if word_lower in months:
            month_name = word_lower
            if i + 1 < len(words):
                next_word = words[i + 1]
                next_word_clean = next_word.lower().rstrip('.,!?;:')
                day_match = re.search(r'^(\d{1,2})(?![0-9])', next_word_clean)
                if day_match:
                    day_num = int(day_match.group(1))
                    if 1 <= day_num <= 31:
                        day_str = day_match.group(1)
                        break
            break
    
    if not month_name or not day_str:
        return None
    
    month = months[month_name]
    day = int(day_str)
    
    if day < 1 or day > 31:
        return None
    
    if post_timestamp is None:
        year = 2000
    else:
        post_date = post_timestamp.to_pydatetime().replace(tzinfo=None)
        post_year = post_date.year
        post_month = post_date.month
        
        if month > post_month:
            year = post_year - 1
        else:
            year = post_year
    
    try:
        return datetime(year=year, month=month, day=day)
    except ValueError:
        return None


def calculate_offset_from_date_pattern3(matched_sentence: str, post_timestamp: pd.Timestamp, phrase: str = "my period on") -> int | None:
    """Calculate offset for pattern_3: "my period on [Month]" or pattern_8: "lmp was [Month]".
    
    Extracts date from the next 3 words after the phrase in the sentence.
    
    Args:
        matched_sentence: Full sentence containing the phrase and date
        post_timestamp: Post's timestamp (pandas Timestamp with timezone)
        phrase: Phrase to search for ("my period on" or "lmp was")
    
    Returns:
        Days between date in sentence and post date, or None if can't parse
    """
    phrase_date = extract_date_after_phrase(matched_sentence, phrase, window_words=3, post_timestamp=post_timestamp)
    
    if phrase_date is None:
        return None
    
    post_date = post_timestamp.to_pydatetime().replace(tzinfo=None)
    
    days_diff = (post_date.date() - phrase_date.date()).days
    
    if days_diff < 0:
        return None
    
    return days_diff


def extract_dpo_from_phrase(matched_phrase: str) -> int | None:
    """Extract DPO number from patterns like 'DPO 8', '8 DPO', 'DPO:8', etc."""
    if pd.isna(matched_phrase):
        return None
    
    patterns = [
        r'\bdpo\s*:?\s*[-\s]*(\d{1,2})\b',
        r'\b(\d{1,2})\s*:?\s*[-\s]*dpo\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, matched_phrase.lower())
        if match:
            return int(match.group(1))
    
    return None


def extract_cd_from_sentence(matched_sentence: str) -> int | None:
    """Extract cycle day number from patterns like 'CD 5', 'i'm on cd 12', etc."""
    if pd.isna(matched_sentence) or not matched_sentence:
        return None
    
    pattern = r'\bcd\s*:?\s*(\d{1,2})\b'
    match = re.search(pattern, matched_sentence.lower())
    if match:
        return int(match.group(1))
    
    return None


def calculate_offset_from_cd1_moon3(
    matched_phrase: str,
    matched_sentence: str | None = None,
    post_timestamp: pd.Timestamp | None = None,
    ovulation_day: int = 14,
) -> int | None:
    """Return offset from cycle day 1 for moon3 patterns.
    
    Priority order (most reliable first):
    - CD X: Extract cycle day from matched_sentence → offset = X - 1
    - LMP was/on: Extract date from matched_sentence
    - DPO X: Days past ovulation → offset = (ovulation_day + X) - 1 (only if CD not available)
    """
    if pd.isna(matched_phrase):
        return None
    
    lowered_phrase = matched_phrase.lower()
    lowered_sentence = matched_sentence.lower() if matched_sentence else ""
    
    # CD patterns: "i'm on cd", "CD 14", etc. - CD is most reliable, so check it first
    # Always check sentence (even if phrase is DPO, sentence might have CD)
    if matched_sentence:
        cd = extract_cd_from_sentence(matched_sentence)
        if cd is not None:
            return cd - 1
    
    
    # DPO patterns: "8 DPO", "DPO 8", etc. - only use if CD not available
    if "dpo" in lowered_phrase:
        dpo = extract_dpo_from_phrase(matched_phrase)
        if dpo is not None:
            cycle_day = ovulation_day + dpo
            return cycle_day - 1
    
    return None


def calculate_offset_by_pattern_type(
    regex_type: str,
    matched_phrase: str,
    matched_sentence: str | None = None,
    post_timestamp: pd.Timestamp | None = None,
    ovulation_day: int = 14,
) -> int | None:
    """Calculate offset based on pattern type - no pattern matching, just extraction.
    
    Args:
        regex_type: Pattern type (pattern_1 through pattern_9)
        matched_phrase: Matched phrase text
        matched_sentence: Full sentence containing the match
        post_timestamp: Post timestamp for date calculations
        ovulation_day: Day of ovulation for DPO calculations (default 14)
    """
    if pd.isna(regex_type) or pd.isna(matched_phrase):
        return None
    
    match regex_type:
        case "pattern_1":
            return calculate_offset_from_simple_marker(matched_phrase)
        
        case "pattern_2":
            return calculate_offset_immediate_start()
        
        case "pattern_3":
            if matched_sentence is not None and post_timestamp is not None:
                return calculate_offset_from_date_pattern3(matched_sentence, post_timestamp)
            return None
        
        case "pattern_4":
            return calculate_offset_from_simple_marker(matched_phrase)
        
        case "pattern_5":
            return calculate_offset_from_explicit_day(matched_phrase)
        
        case "pattern_6":
            return calculate_offset_from_simple_marker(matched_phrase)
        
        case "pattern_8":
            if matched_sentence is not None and post_timestamp is not None:
                phrase = "lmp was on" if "lmp was on" in matched_phrase.lower() else "lmp was"
                return calculate_offset_from_date_pattern3(matched_sentence, post_timestamp, phrase=phrase)
            return None
        
        case "pattern_7" | "pattern_9":
            return calculate_offset_from_cd1_moon3(
                matched_phrase,
                matched_sentence,
                post_timestamp,
                ovulation_day
            )
        
        case _:
            return None


# /////////////////////////////////////////////////////////////////////////////
# OFFSET FOR ARBITRARY POSTS RELATIVE TO CYCLE DAY 1
# /////////////////////////////////////////////////////////////////////////////

def calculate_post_offset_from_anchor(
    post_timestamp: pd.Timestamp | None,
    anchor_timestamp: pd.Timestamp | None,
    anchor_value: int | float | None,
) -> int | None:
    """Calculate time offset for a post relative to anchor.
    
    Generic function that calculates offset by taking the anchor value and adding
    the day difference between post and anchor timestamps.
    
    Works for both:
    - CD patterns: anchor_value is offset_from_cd1 (e.g., 0 for CD1, 2 for CD3)
    - DPO patterns: anchor_value is dpo_days (e.g., 8 for "8 DPO")
    
    Examples:
        - Anchor = 0 (CD1), post is 3 days later      → offset = +3
        - Anchor = 8 (DPO 8), post is 3 days later    → DPO = 11
        - Anchor = 2 (CD3), post is 1 day earlier     → offset = 1
        - Anchor = 8 (DPO 8), post is 2 days earlier  → DPO = 6
    
    The result can be positive or negative, depending on whether the post is after
    or before the anchor reference point.
    
    Args:
        post_timestamp: Post timestamp
        anchor_timestamp: Anchor post timestamp
        anchor_value: Anchor value (offset_from_cd1 or dpo_days)
    
    Returns:
        Calculated offset value or None if inputs are invalid
    """
    if (
        post_timestamp is None
        or anchor_timestamp is None
        or anchor_value is None
        or pd.isna(post_timestamp)
        or pd.isna(anchor_timestamp)
        or pd.isna(anchor_value)
    ):
        return None

    # Ensure we are working with pandas Timestamps
    post_ts = pd.to_datetime(post_timestamp)
    anchor_ts = pd.to_datetime(anchor_timestamp)

    # Integer day difference between post and anchor (can be negative).
    # Use calendar days (normalize to midnight) to avoid time-of-day artifacts.
    days_diff_rounded = (post_ts.normalize() - anchor_ts.normalize()).days

    try:
        anchor_value_int = int(anchor_value)
    except (TypeError, ValueError):
        return None

    return anchor_value_int + days_diff_rounded


def add_offsets_from_anchors(
    df: pd.DataFrame,
    anchors: dict[str, tuple[pd.Timestamp, int]],
    author_col: str = "author",
    timestamp_col: str = "ts_utc",
    pattern_type: str = "cd",
) -> pd.DataFrame:
    """Add offset column (offset_from_cd1 OR dpo_days) using anchor information.
    
    Args:
        df: DataFrame with author and timestamp columns
        anchors: Dictionary mapping user to (anchor_timestamp, anchor_value)
        author_col: Name of author column (default: "author")
        timestamp_col: Name of timestamp column (default: "ts_utc")
        pattern_type: "cd" for cycle day patterns, "dpo" for DPO patterns (default: "cd")
    
    Returns:
        DataFrame with added offset_from_cd1 column (CD) or dpo_days column (DPO)
    """
    df = df.copy()
    column_name = "dpo_days" if pattern_type == "dpo" else "offset_from_cd1"
    
    def compute_offset(row: pd.Series) -> int | None:
        user = str(row[author_col])
        if user not in anchors:
            return None
        
        anchor_ts, anchor_value = anchors[user]
        
        return calculate_post_offset_from_anchor(
            post_timestamp=row[timestamp_col],
            anchor_timestamp=anchor_ts,
            anchor_value=anchor_value,
        )
    
    df[column_name] = df.apply(compute_offset, axis=1)
    
    return df


# /////////////////////////////////////////////////////////////////////////////
# DATAFRAME COLUMN ADDITIONS
# /////////////////////////////////////////////////////////////////////////////

def add_matched_sentences(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Add matched_sentence column based on pattern."""
    df = df.copy()
    df['matched_sentence'] = df['text'].apply(
        lambda x: extract_sentence_with_match(x, pattern)
    )
    return df


def add_matched_phrases(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Add matched_phrase column with exact text that matched pattern."""
    df = df.copy()
    df['matched_phrase'] = df['text'].apply(
        lambda x: extract_matched_phrase(x, pattern)
    )
    return df


def match_patterns_and_identify(text: str, patterns: dict[str, str]) -> tuple[str | None, str | None]:
    """Match text against patterns in order and return which pattern matched and the matched phrase.
    
    Args:
        text: Text to search
        patterns: Dictionary mapping pattern names to regex patterns
        
    Returns:
        Tuple of (pattern_name, matched_phrase) or (None, None) if no match
    """
    if pd.isna(text) or not text:
        return None, None
    
    for pattern_name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return pattern_name, match.group(0)
    
    return None, None


def add_matched_pattern_moon1(df: pd.DataFrame, patterns: dict[str, str]) -> pd.DataFrame:
    """Match moon1 patterns and add regex_type and matched_phrase columns."""
    df = df.copy()
    
    def match_row(text):
        pattern_name, matched_phrase = match_patterns_and_identify(text, patterns)
        return pd.Series({'regex_type': pattern_name, 'matched_phrase': matched_phrase})
    
    result = df['text'].apply(match_row)
    df['regex_type'] = result['regex_type']
    df['matched_phrase'] = result['matched_phrase']
    df['matched_sentence'] = df.apply(
        lambda row: extract_sentence_with_match(row['text'], patterns.get(row['regex_type'], '')) if row['regex_type'] else None,
        axis=1
    )
    return df


def add_matched_pattern_moon2(df: pd.DataFrame, patterns: dict[str, str]) -> pd.DataFrame:
    """Match moon2 patterns in order and add regex_type and matched_phrase columns."""
    df = df.copy()
    
    def match_row(text):
        pattern_name, matched_phrase = match_patterns_and_identify(text, patterns)
        return pd.Series({'regex_type': pattern_name, 'matched_phrase': matched_phrase})
    
    result = df['text'].apply(match_row)
    df['regex_type'] = result['regex_type']
    df['matched_phrase'] = result['matched_phrase']
    df['matched_sentence'] = df.apply(
        lambda row: extract_sentence_with_match(row['text'], patterns.get(row['regex_type'], '')) if row['regex_type'] else None,
        axis=1
    )
    return df


def add_matched_pattern_moon3(df: pd.DataFrame, patterns: dict[str, str]) -> pd.DataFrame:
    """Match moon3 patterns in order and add regex_type and matched_phrase columns."""
    df = df.copy()
    
    def match_row(text):
        pattern_name, matched_phrase = match_patterns_and_identify(text, patterns)
        return pd.Series({'regex_type': pattern_name, 'matched_phrase': matched_phrase})
    
    result = df['text'].apply(match_row)
    df['regex_type'] = result['regex_type']
    df['matched_phrase'] = result['matched_phrase']
    df['matched_sentence'] = df.apply(
        lambda row: extract_sentence_with_match(row['text'], patterns.get(row['regex_type'], '')) if row['regex_type'] else None,
        axis=1
    )
    return df


def add_normalized_phrase(df: pd.DataFrame, source_col: str = "matched_phrase") -> pd.DataFrame:
    """Add matched_phrase_norm column derived from matched_phrase."""
    df = df.copy()
    df["matched_phrase_norm"] = df[source_col].apply(normalize_phrase)
    return df


def add_pattern_type_moon1(df: pd.DataFrame) -> pd.DataFrame:
    """Add regex_type column identifying which moon1 pattern matched."""
    df = df.copy()
    df["regex_type"] = df["matched_phrase"].apply(identify_moon1_pattern_type)
    return df


def add_pattern_type_moon2(df: pd.DataFrame) -> pd.DataFrame:
    """Add regex_type column identifying which moon2 sub-pattern matched."""
    df = df.copy()
    df["regex_type"] = df["matched_phrase"].apply(identify_moon2_pattern_type)
    return df


def add_uncertainty_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_uncertainty flag based on text before matched phrase."""
    df = df.copy()
    df['has_uncertainty'] = df.apply(
        lambda row: check_uncertainty_before_match(
            row['matched_sentence'], row['matched_phrase']
        ), axis=1
    )
    return df


def extract_dpo_days_for_pattern7(
    regex_type: str,
    matched_phrase: str,
    matched_sentence: str | None = None,
) -> int | None:
    """Extract DPO days for pattern7 (days past ovulation).
    
    For pattern7, extracts DPO directly from matched phrase.
    For other patterns, returns None.
    
    Args:
        regex_type: Pattern type (should be "pattern_7" for DPO)
        matched_phrase: Matched phrase text
        matched_sentence: Full sentence (not used for DPO extraction)
    
    Returns:
        DPO value (0-14+) or None if not pattern7 or can't extract
    """
    if pd.isna(regex_type) or regex_type != "pattern_7":
        return None
    
    if pd.isna(matched_phrase):
        return None
    
    return extract_dpo_from_phrase(matched_phrase)


def add_offset_from_cd1_by_pattern(df: pd.DataFrame, ovulation_day: int = 14) -> pd.DataFrame:
    """Attach offset_from_cd1 column based on regex_type pattern.
    
    Also adds dpo_days column for pattern7 users.
    """
    df = df.copy()
    df["offset_from_cd1"] = df.apply(
        lambda row: calculate_offset_by_pattern_type(
            row.get('regex_type'),
            row.get('matched_phrase'),
            row.get('matched_sentence', None),
            row.get('ts_utc', None),
            ovulation_day
        ), axis=1
    )
    
    # Add dpo_days column for pattern7
    df["dpo_days"] = df.apply(
        lambda row: extract_dpo_days_for_pattern7(
            row.get('regex_type'),
            row.get('matched_phrase'),
            row.get('matched_sentence', None),
        ), axis=1
    )
    
    return df




def add_timestamp_columns(
    df: pd.DataFrame,
    utc_col: str = "created_utc",
    tz: str = "UTC",
    add_date_string: bool = False,
) -> pd.DataFrame:
    """Convert unix timestamp to datetime with timezone and formatted string."""
    df = df.copy()
    if utc_col not in df.columns:
        raise KeyError(f"{utc_col} column missing")

    ts_numeric = pd.to_numeric(df[utc_col], errors="coerce")
    df["ts_utc"] = pd.to_datetime(ts_numeric, unit="s", utc=True).dt.tz_convert(tz)
    
    if add_date_string:
        df["ts_date"] = df["ts_utc"].dt.strftime("%Y-%m-%d")
    
    return df


def flag_moderators(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_moderator flag based on author flair and other indicators."""
    df = df.copy()
    
    is_mod = pd.Series([False] * len(df))
    
    if 'author_flair_text' in df.columns:
        flair_mod = df['author_flair_text'].astype(str).str.contains(
            r'\b(mod|moderator|\[m\])\b', case=False, na=False, regex=True
        )
        is_mod = is_mod | flair_mod
    
    if 'distinguished' in df.columns:
        is_mod = is_mod | (df['distinguished'] == 'moderator')
    
    df['is_moderator'] = is_mod
    return df


# /////////////////////////////////////////////////////////////////////////////
# PATTERN STATISTICS
# /////////////////////////////////////////////////////////////////////////////

def count_posts_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Count posts per pattern type and unique users.
    
    Args:
        df: Preprocessed DataFrame with 'regex_type' column
    
    Returns:
        DataFrame with columns: regex_type, count, unique_users
    """
    if 'regex_type' not in df.columns:
        return pd.DataFrame(columns=['regex_type', 'count', 'unique_users'])
    
    counts = df['regex_type'].value_counts().reset_index()
    counts.columns = ['regex_type', 'count']
    
    if 'author' in df.columns:
        unique_users = df.groupby('regex_type')['author'].nunique().reset_index()
        unique_users.columns = ['regex_type', 'unique_users']
        counts = counts.merge(unique_users, on='regex_type', how='left')
    else:
        counts['unique_users'] = None
    
    return counts.sort_values('regex_type')


# /////////////////////////////////////////////////////////////////////////////
# SAMPLING AND VALIDATION
# /////////////////////////////////////////////////////////////////////////////

def sample_posts_by_phrase(
    df: pd.DataFrame,
    phrase_col: str = "matched_phrase_norm",
    sample_size: int = 30,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return up to sample_size random rows per phrase (uniform sampling).
    
    Args:
        df: DataFrame to sample from
        phrase_col: Column to group by for sampling
        sample_size: Number of rows to sample per phrase
        random_state: Random seed (should be passed from config for reproducibility)
    
    Returns:
        DataFrame with sampled rows
    """
    if phrase_col not in df.columns:
        raise KeyError(f"{phrase_col} column missing")

    samples: list[pd.DataFrame] = []

    for phrase, group in df.groupby(phrase_col):
        if pd.isna(phrase):
            continue

        selected = group.sample(
            n=min(len(group), sample_size),
            random_state=random_state,
        ).sort_index().reset_index(drop=True)

        selected["sample_index"] = range(1, len(selected) + 1)
        selected["sample_size_target"] = sample_size
        samples.append(selected)

    if not samples:
        return pd.DataFrame()

    return pd.concat(samples, ignore_index=True)
