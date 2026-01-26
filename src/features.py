"""Functions for computing linguistic features on text data."""
from __future__ import annotations

import pandas as pd
import numpy as np
import re

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: tqdm is optional, just use regular iteration
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc)
        return iterable

from src.attributes import (
    syntactic_complexity_batch,
    mattr_score,
    spelling_error_fraction,
    cohesion_score,
)
from src.analysis import normalize_values


def compute_textblob_sentiment(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute TextBlob sentiment features.
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added columns:
        - textblob_polarity: -1 (negative) to +1 (positive)
        - textblob_subjectivity: 0 (objective) to 1 (subjective)
        - textblob_intensity: absolute value of polarity
    """
    print(f"Computing TextBlob sentiment on column: {text_column}")
    
    try:
        from textblob import TextBlob
    except ImportError:
        print("✗ TextBlob not available, skipping")
        df['textblob_polarity'] = 0.0
        df['textblob_subjectivity'] = 0.0
        df['textblob_intensity'] = 0.0
        return df
    
    def get_sentiment(text):
        try:
            if pd.isna(text) or len(str(text).strip()) == 0:
                return {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'intensity': abs(blob.sentiment.polarity)
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}
    
    try:
        tqdm.pandas(desc="Processing TextBlob")
        sentiments = df[text_column].progress_apply(get_sentiment)
    except (AttributeError, ImportError):
        sentiments = df[text_column].apply(get_sentiment)
    
    df['textblob_polarity'] = sentiments.apply(lambda x: x['polarity'])
    df['textblob_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
    df['textblob_intensity'] = sentiments.apply(lambda x: x['intensity'])
    
    print(f"✓ TextBlob computed: mean polarity = {df['textblob_polarity'].mean():.3f}")
    return df


def compute_vader_sentiment(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute VADER sentiment features.
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added columns:
        - sentiment_negative: proportion of negative words (0-1)
        - sentiment_positive: proportion of positive words (0-1)
        - sentiment_compound: overall sentiment (-1 to +1)
    """
    print(f"Computing VADER sentiment on column: {text_column}")
    
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"✗ Could not load VADER: {e}")
        df['sentiment_negative'] = 0
        df['sentiment_positive'] = 0
        df['sentiment_compound'] = 0
        return df
    
    def get_vader_scores(text):
        """Helper function that returns a Series with neg, pos, compound."""
        try:
            if pd.isna(text) or len(str(text).strip()) == 0:
                return pd.Series({'neg': 0, 'pos': 0, 'compound': 0})
            scores = sia.polarity_scores(str(text))
            return pd.Series({'neg': scores['neg'], 'pos': scores['pos'], 'compound': scores['compound']})
        except:
            return pd.Series({'neg': 0, 'pos': 0, 'compound': 0})
    
    # Enable tqdm progress bar for pandas apply operations
    try:
        tqdm.pandas(desc="Processing VADER")
        vader_df = df[text_column].progress_apply(get_vader_scores)
    except (AttributeError, ImportError):
        # Fallback if tqdm.pandas() not available
        vader_df = df[text_column].apply(get_vader_scores)
    
    df['sentiment_negative'] = vader_df['neg']
    df['sentiment_positive'] = vader_df['pos']
    df['sentiment_compound'] = vader_df['compound']
    
    print(f"✓ VADER computed: mean compound = {df['sentiment_compound'].mean():.3f}")
    return df


def compute_syntactic_complexity(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute syntactic complexity (slow spaCy operation).
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added column:
        - syntactic_complexity: subordinate clauses per sentence
    """
    print(f"Computing syntactic complexity on {len(df):,} texts...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_md')
        print("✓ spaCy loaded")
        
        df['syntactic_complexity'] = syntactic_complexity_batch(
            df[text_column].fillna('').tolist(), nlp
        )
        print(f"✓ Syntactic complexity computed: mean = {df['syntactic_complexity'].mean():.3f}")
    except Exception as e:
        print(f"✗ spaCy error: {e}")
        print("  Setting syntactic_complexity=0")
        df['syntactic_complexity'] = 0.0
    
    return df


def compute_cohesion(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute cohesion (very slow spaCy operation).
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added column:
        - cohesion: semantic similarity (0-1)
    """
    print(f"Computing cohesion on {len(df):,} texts (this is VERY SLOW)...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_md')
        print("✓ spaCy loaded")
        
        cohesion_scores = []
        for text in tqdm(df[text_column].fillna(''), 
                        total=len(df), 
                        desc="   Processing cohesion",
                        unit="text"):
            cohesion_scores.append(cohesion_score(text, nlp))
        df['cohesion'] = cohesion_scores
        df['cohesion'] = df['cohesion'].replace([None], np.nan)
        print(f"✓ Cohesion computed: mean = {df['cohesion'].mean():.3f}")
    except Exception as e:
        print(f"✗ spaCy error: {e}")
        print("  Setting cohesion=NaN")
        df['cohesion'] = np.nan
    
    return df


def compute_basic_linguistic_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute fast linguistic features (word count, avg length, etc.).
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added columns:
        - word_count: total words
        - avg_word_length: characters per word
        - avg_words_per_sentence: words per sentence
        - unique_word_fraction: lexical diversity (0-1)
        - flesch_kincaid: reading grade level
    """
    print(f"Computing basic linguistic features on column: {text_column}")
    
    try:
        import textstat
    except:
        textstat = None
    
    print("\n1. Word count...")
    try:
        tqdm.pandas(desc="   Processing word_count")
        df['word_count'] = df[text_column].fillna('').progress_apply(lambda x: len(str(x).split()))
    except (AttributeError, ImportError):
        df['word_count'] = df[text_column].fillna('').apply(lambda x: len(str(x).split()))
    print(f"   ✓ Mean: {df['word_count'].mean():.0f}")
    
    print("\n2. Average word length...")
    def avg_word_len(text):
        words = str(text).split()
        return np.mean([len(w) for w in words]) if words else 0.0
    try:
        tqdm.pandas(desc="   Processing avg_word_length")
        df['avg_word_length'] = df[text_column].fillna('').progress_apply(avg_word_len)
    except (AttributeError, ImportError):
        df['avg_word_length'] = df[text_column].fillna('').apply(avg_word_len)
    print(f"   ✓ Mean: {df['avg_word_length'].mean():.2f} chars")
    
    print("\n3. Average words per sentence...")
    def words_per_sentence(text):
        text_str = str(text)
        if not text_str.strip():
            return 0.0
        sentences = [s.strip() for s in re.split(r'[.!?]', text_str) if s.strip()]
        if not sentences:
            return 0.0
        return np.mean([len(s.split()) for s in sentences])
    try:
        tqdm.pandas(desc="   Processing avg_words_per_sentence")
        df['avg_words_per_sentence'] = df[text_column].fillna('').progress_apply(words_per_sentence)
    except (AttributeError, ImportError):
        df['avg_words_per_sentence'] = df[text_column].fillna('').apply(words_per_sentence)
    print(f"   ✓ Mean: {df['avg_words_per_sentence'].mean():.1f}")
    
    print("\n4. Unique word fraction...")
    def unique_fraction(text):
        words = str(text).lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    try:
        tqdm.pandas(desc="   Processing unique_word_fraction")
        df['unique_word_fraction'] = df[text_column].fillna('').progress_apply(unique_fraction)
    except (AttributeError, ImportError):
        df['unique_word_fraction'] = df[text_column].fillna('').apply(unique_fraction)
    print(f"   ✓ Mean: {df['unique_word_fraction'].mean():.3f}")
    
    print("\n5. Flesch-Kincaid grade level...")
    if textstat:
        try:
            tqdm.pandas(desc="   Processing flesch_kincaid")
            df['flesch_kincaid'] = df[text_column].fillna('').progress_apply(
                lambda x: textstat.flesch_kincaid_grade(str(x)) if x else 0.0
            )
        except (AttributeError, ImportError):
            df['flesch_kincaid'] = df[text_column].fillna('').apply(
                lambda x: textstat.flesch_kincaid_grade(str(x)) if x else 0.0
            )
        print(f"   ✓ Mean: {df['flesch_kincaid'].mean():.1f}")
    else:
        df['flesch_kincaid'] = 0.0
    
    print("✓ Basic linguistic features computed")
    return df


def compute_advanced_linguistic_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute advanced linguistic features (MATTR, spelling - slow).
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added columns:
        - mattr: lexical richness (0-1)
        - spelling_error_fraction: fraction of misspelled words (0-1)
    """
    print(f"Computing advanced linguistic features on column: {text_column}")
    
    try:
        from lexicalrichness import LexicalRichness
    except:
        LexicalRichness = None
    
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
    except:
        spell = None
    
    print("\n1. MATTR (lexical richness - slow)...")
    if LexicalRichness:
        mattr_scores = []
        for text in tqdm(df[text_column].fillna(''), desc="   Processing MATTR"):
            mattr_scores.append(mattr_score(text))
        df['mattr'] = mattr_scores
        df['mattr'] = df['mattr'].replace([None], np.nan)
        print(f"   ✓ Mean: {df['mattr'].mean():.3f}")
    else:
        print("   ✗ LexicalRichness not available, setting mattr=NaN")
        df['mattr'] = np.nan
    
    print("\n2. Spelling error fraction (slow)...")
    if spell:
        try:
            tqdm.pandas(desc="   Processing spelling_error_fraction")
            df['spelling_error_fraction'] = df[text_column].fillna('').progress_apply(
                lambda x: spelling_error_fraction(str(x), spell)
            )
        except (AttributeError, ImportError):
            df['spelling_error_fraction'] = df[text_column].fillna('').apply(
                lambda x: spelling_error_fraction(str(x), spell)
            )
        print(f"   ✓ Mean: {df['spelling_error_fraction'].mean():.3f}")
    else:
        print("   ✗ SpellChecker not available, setting spelling_error_fraction=0")
        df['spelling_error_fraction'] = 0.0
    
    print("✓ Advanced linguistic features computed")
    return df


def compute_linguistic_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute comprehensive linguistic features (legacy combined function).
    
    This is the original monolithic function, kept for backward compatibility.
    For better checkpointing, use the modular functions separately:
    - compute_syntactic_complexity()
    - compute_cohesion()
    - compute_basic_linguistic_features()
    - compute_advanced_linguistic_features()
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with added columns:
        - syntactic_complexity: subordinate clauses per sentence
        - cohesion: semantic similarity (0-1)
        - word_count: total words
        - avg_word_length: characters per word
        - avg_words_per_sentence: words per sentence
        - unique_word_fraction: lexical diversity (0-1)
        - flesch_kincaid: reading grade level
        - mattr: lexical richness (0-1)
        - spelling_error_fraction: fraction of misspelled words (0-1)
    """
    print(f"Computing linguistic features on column: {text_column}")
    print("  ⚠️  Using monolithic function - consider using modular functions for checkpointing")
    
    df = compute_syntactic_complexity(df, text_column)
    df = compute_cohesion(df, text_column)
    df = compute_basic_linguistic_features(df, text_column)
    df = compute_advanced_linguistic_features(df, text_column)
    
    print("\n✓ All linguistic features computed")
    return df


def compute_all_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute all features: TextBlob + VADER + Linguistic.
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text to analyze
        
    Returns:
        DataFrame with all feature columns added
    """
    print("="*60)
    print("COMPUTING ALL FEATURES")
    print("="*60)
    print()
    
    print("PART 1: TextBlob Sentiment")
    print("-"*60)
    df = compute_textblob_sentiment(df, text_column)
    print()
    
    print("PART 2: VADER Sentiment")
    print("-"*60)
    df = compute_vader_sentiment(df, text_column)
    print()
    
    print("PART 3: Linguistic Features")
    print("-"*60)
    df = compute_linguistic_features(df, text_column)
    print()
    
    print("="*60)
    print("✓ ALL FEATURES COMPUTED")
    print("="*60)
    return df


def normalize_features_per_user(
    df: pd.DataFrame,
    feature_cols: list[str],
    user_col: str = "author",
) -> pd.DataFrame:
    """Normalize features per user using zscore method.
    
    For each user, normalizes specified features independently.
    Adds columns: {feature}_zscore for each feature.
    
    Args:
        df: DataFrame with features to normalize
        feature_cols: List of feature column names to normalize
        user_col: Column name for user identifier (default: "author")
        
    Returns:
        DataFrame with added normalized feature columns
    """
    df = df.copy()
    
    print(f"Normalizing {len(feature_cols)} features per user...")
    
    # Build all normalized columns at once to avoid DataFrame fragmentation
    normalized_data = {}
    
    for feature in feature_cols:
        if feature not in df.columns:
            print(f"  Warning: {feature} not found, skipping")
            continue
        
        print(f"  Processing {feature}...")
        
        zscore_col = f"{feature}_zscore"
        
        # Initialize with NaN
        zscore_values = np.full(len(df), np.nan, dtype=np.float64)
        
        for user, group in df.groupby(user_col):
            values = group[feature].values
            
            if len(values) < 2:
                continue
            
            if np.std(values) < 1e-10:
                continue
            
            idx = group.index
            zscore_values[idx] = normalize_values(values, method="zscore")
        
        # Store in dictionary to add all at once later
        normalized_data[zscore_col] = zscore_values
        
        valid_zscore = (~np.isnan(zscore_values)).sum()
        print(f"    ✓ {zscore_col}: {valid_zscore} valid entries")
    
    # Add all normalized columns at once using pd.concat to avoid fragmentation
    if normalized_data:
        normalized_df = pd.DataFrame(normalized_data, index=df.index)
        df = pd.concat([df, normalized_df], axis=1)
    
    print("✓ Per-user normalization complete")
    return df

