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


def compute_linguistic_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Compute comprehensive linguistic features.
    
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
    
    print("Loading NLP models...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_md')
        print("✓ spaCy loaded")
    except:
        print("✗ spaCy not available")
        nlp = None
    
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        print("✓ Spell checker loaded")
    except:
        print("✗ Spell checker not available")
        spell = None
    
    try:
        import textstat
        print("✓ Textstat loaded")
    except:
        print("✗ Textstat not available")
        textstat = None
    
    try:
        from lexicalrichness import LexicalRichness
        print("✓ Lexical richness loaded")
    except:
        print("✗ Lexical richness not available")
        LexicalRichness = None
    
    print("\n1. Syntactic complexity...")
    if nlp:
        df['syntactic_complexity'] = syntactic_complexity_batch(
            df[text_column].fillna('').tolist(), nlp
        )
        print(f"   ✓ Mean: {df['syntactic_complexity'].mean():.3f}")
    else:
        df['syntactic_complexity'] = 0.0
    
    print("\n2. Cohesion (slow)...")
    if nlp:
        cohesion_scores = []
        for text in tqdm(df[text_column].fillna(''), desc="   Processing"):
            cohesion_scores.append(cohesion_score(text, nlp))
        df['cohesion'] = cohesion_scores
        print(f"   ✓ Mean: {df['cohesion'].mean():.3f}")
    else:
        df['cohesion'] = None
    
    print("\n3. Word count...")
    df['word_count'] = df[text_column].fillna('').apply(lambda x: len(str(x).split()))
    print(f"   ✓ Mean: {df['word_count'].mean():.0f}")
    
    print("\n4. Average word length...")
    def avg_word_len(text):
        words = str(text).split()
        return np.mean([len(w) for w in words]) if words else 0.0
    df['avg_word_length'] = df[text_column].fillna('').apply(avg_word_len)
    print(f"   ✓ Mean: {df['avg_word_length'].mean():.2f} chars")
    
    print("\n5. Average words per sentence...")
    def words_per_sentence(text):
        text_str = str(text)
        if not text_str.strip():
            return 0.0
        sentences = [s.strip() for s in re.split(r'[.!?]', text_str) if s.strip()]
        if not sentences:
            return 0.0
        return np.mean([len(s.split()) for s in sentences])
    df['avg_words_per_sentence'] = df[text_column].fillna('').apply(words_per_sentence)
    print(f"   ✓ Mean: {df['avg_words_per_sentence'].mean():.1f}")
    
    print("\n6. Unique word fraction...")
    def unique_fraction(text):
        words = str(text).lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    df['unique_word_fraction'] = df[text_column].fillna('').apply(unique_fraction)
    print(f"   ✓ Mean: {df['unique_word_fraction'].mean():.3f}")
    
    print("\n7. Flesch-Kincaid grade level...")
    if textstat:
        df['flesch_kincaid'] = df[text_column].fillna('').apply(
            lambda x: textstat.flesch_kincaid_grade(str(x)) if x else 0.0
        )
        print(f"   ✓ Mean: {df['flesch_kincaid'].mean():.1f}")
    else:
        df['flesch_kincaid'] = 0.0
    
    print("\n8. MATTR (lexical richness)...")
    if LexicalRichness:
        mattr_scores = []
        for text in tqdm(df[text_column].fillna(''), desc="   Processing"):
            mattr_scores.append(mattr_score(text))
        df['mattr'] = mattr_scores
        print(f"   ✓ Mean: {df['mattr'].mean():.3f}")
    else:
        df['mattr'] = None
    
    print("\n9. Spelling error fraction...")
    if spell:
        df['spelling_error_fraction'] = df[text_column].fillna('').apply(
            lambda x: spelling_error_fraction(str(x), spell)
        )
        print(f"   ✓ Mean: {df['spelling_error_fraction'].mean():.3f}")
    else:
        df['spelling_error_fraction'] = 0.0
    
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
    """Normalize features per user using zscore and minmax methods.
    
    For each user, normalizes specified features independently.
    Adds columns: {feature}_zscore, {feature}_minmax for each feature.
    
    Args:
        df: DataFrame with features to normalize
        feature_cols: List of feature column names to normalize
        user_col: Column name for user identifier (default: "author")
        
    Returns:
        DataFrame with added normalized feature columns
    """
    df = df.copy()
    
    print(f"Normalizing {len(feature_cols)} features per user...")
    
    for feature in feature_cols:
        if feature not in df.columns:
            print(f"  Warning: {feature} not found, skipping")
            continue
        
        print(f"  Processing {feature}...")
        
        zscore_col = f"{feature}_zscore"
        minmax_col = f"{feature}_minmax"
        
        df[zscore_col] = np.nan
        df[minmax_col] = np.nan
        
        for user, group in df.groupby(user_col):
            values = group[feature].values
            
            if len(values) < 2:
                continue
            
            if np.std(values) < 1e-10:
                continue
            
            idx = group.index
            df.loc[idx, zscore_col] = normalize_values(values, method="zscore")
            df.loc[idx, minmax_col] = normalize_values(values, method="minmax")
        
        valid_zscore = df[zscore_col].notna().sum()
        valid_minmax = df[minmax_col].notna().sum()
        print(f"    ✓ {zscore_col}: {valid_zscore} valid entries")
        print(f"    ✓ {minmax_col}: {valid_minmax} valid entries")
    
    print("✓ Per-user normalization complete")
    return df

