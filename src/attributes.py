"""Utility functions for computing linguistic attributes."""
from __future__ import annotations

import numpy as np

try:
    from lexicalrichness import LexicalRichness
except ImportError:
    LexicalRichness = None


def syntactic_complexity_batch(texts, nlp, batch_size=300, n_process=1):
    """Compute syntactic complexity for a batch of texts.
    
    Args:
        texts: List of text strings
        nlp: spaCy language model
        batch_size: Batch size for processing
        n_process: Number of processes (1 for Windows compatibility)
        
    Returns:
        List of complexity scores
    """
    if nlp is None:
        return [0.0] * len(texts)
    
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        sents = list(doc.sents)
        if len(sents) == 0:
            results.append(0.0)
            continue
        complexity = sum(1 for token in doc if token.dep_ in ['mark', 'cc']) / len(sents)
        results.append(complexity)
    return results


def mattr_score(text, window_size=50):
    """Compute MATTR (lexical diversity) score.
    
    Args:
        text: Text string
        window_size: Window size for moving average type-token ratio
        
    Returns:
        MATTR score (float) or None if text is invalid
    """
    if LexicalRichness is None:
        return None
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    lex = LexicalRichness(text)
    n_words = lex.words
    
    if n_words == 0:
        return None
    
    if n_words < window_size:
        return lex.mattr(window_size=n_words)
    
    return lex.mattr(window_size=window_size)


def spelling_error_fraction(text, spell):
    """Compute fraction of misspelled words.
    
    Args:
        text: Text string
        spell: SpellChecker instance
        
    Returns:
        Fraction of misspelled words (0-1)
    """
    if spell is None:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    misspelled = spell.unknown(words)
    return len(misspelled) / len(words)


def cohesion_score(text, nlp):
    """Compute semantic cohesion between sentences.
    
    Args:
        text: Text string
        nlp: spaCy language model
        
    Returns:
        Average similarity score between adjacent sentences (float) or None
    """
    if nlp is None:
        return None
    
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    doc = nlp(text)
    sents = list(doc.sents)
    if len(sents) < 2:
        return None

    sims = [
        s1.similarity(s2)
        for s1, s2 in zip(sents[:-1], sents[1:])
        if s1.vector_norm and s2.vector_norm
    ]

    return float(np.mean(sims)) if sims else None

