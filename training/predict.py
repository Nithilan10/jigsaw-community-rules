# ================================
# Kaggle submission script (offline)
# Self-contained: long preprocess + model/inference
# Readability now works w/o textstat
# ================================

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# BEGIN: INLINE COPY OF YOUR LONG preprocess.py (same as original)
# ONLY CHANGE: textstat import is optional + offline readability fallback
# --------------------------------------------------------------------

import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, normalize, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy.spatial.distance import cosine
import numpy as np
from typing import Tuple, Dict, Any

# spaCy is only used if enable_spacy=True later; keep the import to match your file.
import spacy

# ---- textstat import is optional now ----
try:
    import textstat
    _HAS_TEXTSTAT = True
except Exception:
    _HAS_TEXTSTAT = False

from collections import Counter
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

LEXICAL_CUES = r'\b(you should|you must|i suggest|my advice|best way is to)\b'
SEMANTIC_KEYWORDS = r'\b(sue|lawyer|court|filing|testimony|statute|jurisdiction|legal advice)\b'
PROMO_CUES = r'\b(free|limited|giveaway|discount|click here|watch now|c0mpanyname)\b'
OBFUSCATED_NAMES = r'\b(gamify|c0in|fr3e|cIick|Iink)\b' 

# --- Enhanced Text Processing Functions (unchanged) ---
def _clean_and_normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    text = re.sub(r'[.]{3,}', '...', text)
    return text.strip()

def _get_exclamation_frequency(comment: str) -> float:
    n = len(comment)
    return 0.0 if n == 0 else comment.count('!') / n

def _check_legal_advice_interaction(comment: str) -> int:
    text = comment.lower()
    return 1 if (re.search(LEXICAL_CUES, text) and re.search(SEMANTIC_KEYWORDS, text)) else 0

def _calculate_promo_persuasion_feature(comment: str) -> int:
    text = comment.lower()
    return 1 if re.search(PROMO_CUES, text) or re.search(OBFUSCATED_NAMES, text) else 0

# --- Stylometrics (unchanged) ---
def extract_stylometric_features(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {'exclamation_ratio':0.0,'question_ratio':0.0,'period_ratio':0.0,
                'uppercase_ratio':0.0,'title_case_ratio':0.0,'short_word_ratio':0.0,
                'long_word_ratio':0.0,'avg_sentence_length':0.0,'punctuation_density':0.0,
                'capitalization_ratio':0.0}
    feats = {}
    n = len(text)
    feats['exclamation_ratio'] = text.count('!')/n
    feats['question_ratio']   = text.count('?')/n
    feats['period_ratio']     = text.count('.')/n
    feats['punctuation_density'] = sum(1 for c in text if c in '!?.,;:')/n
    feats['uppercase_ratio']  = sum(1 for c in text if c.isupper())/n
    words = text.split()
    if words:
        feats['title_case_ratio'] = sum(1 for w in words if w.istitle())/len(words)
        feats['capitalization_ratio'] = sum(1 for w in words if any(c.isupper() for c in w))/len(words)
        feats['short_word_ratio'] = sum(1 for w in words if len(w)<=3)/len(words)
        feats['long_word_ratio']  = sum(1 for w in words if len(w)>=7)/len(words)
    else:
        feats['title_case_ratio']=feats['capitalization_ratio']=feats['short_word_ratio']=feats['long_word_ratio']=0.0
    sents = [s.strip() for s in text.split('.') if s.strip()]
    feats['avg_sentence_length'] = (sum(len(s.split()) for s in sents)/len(sents)) if sents else 0.0
    return feats

def calculate_group_stylometric_features(texts: list) -> dict:
    if not texts: return get_empty_group_features()
    all_feats = [extract_stylometric_features(t) for t in texts if isinstance(t,str) and t.strip()]
    if not all_feats: return get_empty_group_features()
    out = {}
    keys = list(all_feats[0].keys())
    for k in keys:
        vals = [f[k] for f in all_feats]
        out[f'group_{k}_mean']=np.mean(vals)
        out[f'group_{k}_std'] =np.std(vals) if len(vals)>1 else 0.0
        out[f'group_{k}_max'] =np.max(vals)
        out[f'group_{k}_min'] =np.min(vals)
    return out

def get_empty_group_features()->dict:
    base=['exclamation_ratio','question_ratio','period_ratio','uppercase_ratio','title_case_ratio',
          'short_word_ratio','long_word_ratio','avg_sentence_length','punctuation_density','capitalization_ratio']
    return {f'group_{b}_{suf}':0.0 for b in base for suf in ['mean','std','max','min']}

def create_comparison_features(pos:dict,neg:dict)->dict:
    comp={}
    for k in pos.keys():
        if k.startswith('group_') and k.endswith('_mean'):
            base=k.replace('group_','').replace('_mean','')
            p=pos[k]; n=neg[k]
            comp[f'{base}_violation_vs_safe_diff']=p-n
            comp[f'{base}_violation_vs_safe_ratio']= (p/n) if n!=0 else 1.0
            std=pos.get(f'group_{base}_std',1.0)
            comp[f'{base}_violation_zscore']= (p-n)/std if std!=0 else 0.0
    return comp

def calculate_context_aware_stylometric_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating context-aware stylometric features...")
    
    # Check if required columns exist
    required_cols = ['subreddit', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Required columns for context-aware stylometric features not found. Using generic features.")
        rows = []
        for _ in range(len(df)):
            rows.append(get_generic_comparison_features())
        df = pd.concat([df, pd.DataFrame(rows)], axis=1)
        print(f"Added {len(rows[0]) if rows else 0} generic stylometric features")
        return df
    
    buckets={}
    for _,row in df.iterrows():
        key=f"{row['subreddit']}_{row['rule']}"
        buckets.setdefault(key,{'positive':[],'negative':[]})
        for c in ['positive_example_1','positive_example_2']:
            if c in row and pd.notna(row[c]): buckets[key]['positive'].append(str(row[c]))
        for c in ['negative_example_1','negative_example_2']:
            if c in row and pd.notna(row[c]): buckets[key]['negative'].append(str(row[c]))
    cache={}
    for key,ex in buckets.items():
        cache[key]={'positive':calculate_group_stylometric_features(ex['positive']),
                    'negative':calculate_group_stylometric_features(ex['negative'])}
    rows=[]
    for _,row in df.iterrows():
        key=f"{row['subreddit']}_{row['rule']}"
        if key in cache:
            rows.append(create_comparison_features(cache[key]['positive'],cache[key]['negative']))
        else:
            rows.append(get_generic_comparison_features())
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    print(f"Added {len(rows[0]) if rows else 0} context-aware stylometric features")
    return df

def get_generic_comparison_features()->dict:
    base=['exclamation_ratio','question_ratio','period_ratio','uppercase_ratio','title_case_ratio',
          'short_word_ratio','long_word_ratio','avg_sentence_length','punctuation_density','capitalization_ratio']
    out={}
    for b in base:
        out[f'{b}_violation_vs_safe_diff']=0.0
        out[f'{b}_violation_vs_safe_ratio']=1.0
        out[f'{b}_violation_zscore']=0.0
    return out

# ---------------------------
# READABILITY (UPDATED PART)
# ---------------------------
_vowel_re = re.compile(r'[aeiouy]+', re.I)

def _estimate_syllables(word: str) -> int:
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w: return 0
    groups = _vowel_re.findall(w)
    count = len(groups)
    # silent 'e'
    if w.endswith('e') and not w.endswith(('le','ue')) and count>1:
        count -= 1
    return max(1, count)

def _readability_fallback(text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        return {
            'flesch_kincaid':0.0,'gunning_fog':0.0,'flesch_reading_ease':0.0,'smog_index':0.0,
            'avg_sentence_length_readability':0.0,'avg_syllables_per_word':0.0
        }
    # sentence split (simple)
    sent_splits = re.split(r'[.!?]+', text)
    sentences = [s for s in sent_splits if s.strip()]
    S = max(1, len(sentences))
    words = re.findall(r"[A-Za-z']+", text)
    W = max(1, len(words))
    syllables = sum(_estimate_syllables(w) for w in words)
    ASL = W / S
    ASW = syllables / W
    # Flesch Reading Ease
    FRE = 206.835 - 1.015*ASL - 84.6*ASW
    # Flesch-Kincaid Grade
    FK = 0.39*ASL + 11.8*ASW - 15.59
    # Complex (>=3 syllables)
    complex_words = sum(1 for w in words if _estimate_syllables(w) >= 3)
    pct_complex = (complex_words / W) * 100.0
    # Gunning Fog
    GF = 0.4 * (ASL + pct_complex)
    # SMOG (needs polysyllables & sentences)
    try:
        SMOG = 1.0430 * np.sqrt(complex_words * (30.0 / S)) + 3.1291 if S > 0 else 0.0
    except Exception:
        SMOG = 0.0
    return {
        'flesch_kincaid': float(FK),
        'gunning_fog': float(GF),
        'flesch_reading_ease': float(FRE),
        'smog_index': float(SMOG),
        'avg_sentence_length_readability': float(ASL),
        'avg_syllables_per_word': float(ASW)
    }

def extract_readability_features(text: str) -> dict:
    """
    Updated: uses textstat if available; otherwise offline heuristic fallback.
    Feature keys stay the same.
    """
    if _HAS_TEXTSTAT:
        try:
            feats = {}
            try: feats['flesch_kincaid'] = textstat.flesch_kincaid_grade(text)
            except AttributeError:
                try: feats['flesch_kincaid'] = textstat.flesch_kincaid(text)
                except AttributeError: feats['flesch_kincaid'] = 0.0
            try: feats['gunning_fog'] = textstat.gunning_fog(text)
            except AttributeError: feats['gunning_fog'] = 0.0
            try: feats['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            except AttributeError: feats['flesch_reading_ease'] = 0.0
            try: feats['smog_index'] = textstat.smog_index(text)
            except AttributeError: feats['smog_index'] = 0.0
            try: feats['avg_sentence_length_readability'] = textstat.avg_sentence_length(text)
            except AttributeError: feats['avg_sentence_length_readability'] = 0.0
            try: feats['avg_syllables_per_word'] = textstat.avg_syllables_per_word(text)
            except AttributeError: feats['avg_syllables_per_word'] = 0.0
            return feats
        except Exception:
            # fall back if textstat throws any runtime error
            return _readability_fallback(text)
    else:
        return _readability_fallback(text)

# --- Lexical diversity (unchanged) ---
def extract_lexical_diversity_features(text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        return {'type_token_ratio':0.0,'lexical_diversity':0.0,'avg_word_length_lexical':0.0,
                'vocabulary_richness':0.0,'most_common_word_ratio':0.0}
    try:
        words = [w.lower() for w in text.split() if w]
        if not words: 
            return {'type_token_ratio':0.0,'lexical_diversity':0.0,'avg_word_length_lexical':0.0,
                    'vocabulary_richness':0.0,'most_common_word_ratio':0.0}
        uniq=set(words)
        ttr = len(uniq)/len(words)
        avg_len = sum(len(w) for w in words)/len(words)
        vocab_rich = (len(uniq)/len(words))*100.0
        freq = Counter(words)
        mcr = freq.most_common(1)[0][1] / len(words)
        return {'type_token_ratio':ttr,'lexical_diversity':ttr,'avg_word_length_lexical':avg_len,
                'vocabulary_richness':vocab_rich,'most_common_word_ratio':mcr}
    except Exception:
        return {'type_token_ratio':0.0,'lexical_diversity':0.0,'avg_word_length_lexical':0.0,
                'vocabulary_richness':0.0,'most_common_word_ratio':0.0}

def extract_pos_features(text: str, nlp) -> dict:
    """
    Extract Part-of-Speech (POS) tag features using spaCy
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_pos_features()
    
    try:
        doc = nlp(text)
        pos_counts = {}
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        total_tokens = len([token for token in doc if not token.is_space and not token.is_punct])
        
        if total_tokens == 0:
            return get_empty_pos_features()
        
        pos_features = {}
        for pos, count in pos_counts.items():
            pos_features[f'pos_{pos.lower()}_ratio'] = count / total_tokens
        
        # Fill missing POS tags with 0
        all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
        for pos in all_pos_tags:
            if f'pos_{pos.lower()}_ratio' not in pos_features:
                pos_features[f'pos_{pos.lower()}_ratio'] = 0.0
        
        return pos_features
        
    except Exception as e:
        print(f"Error in POS feature extraction: {e}")
        return get_empty_pos_features()

def get_empty_pos_features() -> dict:
    """Return empty POS features"""
    all_pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
    return {f'pos_{pos.lower()}_ratio': 0.0 for pos in all_pos_tags}

def extract_dependency_features(text: str, nlp) -> dict:
    """
    Extract dependency parsing features using spaCy
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_dependency_features()
    
    try:
        doc = nlp(text)
        dep_counts = {}
        
        for token in doc:
            if not token.is_space:
                dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
        
        total_tokens = len([token for token in doc if not token.is_space])
        
        if total_tokens == 0:
            return get_empty_dependency_features()
        
        dep_features = {}
        for dep, count in dep_counts.items():
            dep_features[f'dep_{dep.lower()}_ratio'] = count / total_tokens
        
        # Specific dependency patterns for legal/promotional content
        dep_features['has_imperative'] = any(token.dep_ == 'ROOT' and token.tag_ == 'VB' for token in doc)
        dep_features['has_conditional'] = any(token.dep_ == 'mark' for token in doc)
        dep_features['has_negation'] = any(token.dep_ == 'neg' for token in doc)
        dep_features['has_auxiliary'] = any(token.dep_ == 'aux' for token in doc)
        
        return dep_features
        
    except Exception as e:
        print(f"Error in dependency feature extraction: {e}")
        return get_empty_dependency_features()

def get_empty_dependency_features() -> dict:
    """Return empty dependency features"""
    return {
        'has_imperative': False,
        'has_conditional': False,
        'has_negation': False,
        'has_auxiliary': False
    }

def calculate_advanced_text_features(df: pd.DataFrame, enable_spacy: bool = False) -> pd.DataFrame:
    print("Calculating advanced text features...")
    nlp=None
    if enable_spacy:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except Exception:
            print("Warning: spaCy model not available; using fallbacks.")
            nlp=None
    rows=[]
    for idx,row in df.iterrows():
        text = row['comment_text']
        feats = {}
        
        # POS and dependency features
        if nlp is not None:
            feats.update(extract_pos_features(text, nlp))
            feats.update(extract_dependency_features(text, nlp))
        else:
            # Fallback to empty features
            feats.update(get_empty_pos_features())
            feats.update(get_empty_dependency_features())
        
        # Readability + lexical diversity
        feats.update(extract_readability_features(text))
        feats.update(extract_lexical_diversity_features(text))
        rows.append(feats)
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    print(f"Added {len(rows[0]) if rows else 0} advanced text features")
    return df

# --- Domain-Specific Features ---

# Legal and Brand Recognition Patterns
LEGAL_TERMS = [
    'lawsuit', 'litigation', 'court', 'judge', 'jury', 'trial', 'verdict', 'settlement',
    'plaintiff', 'defendant', 'attorney', 'lawyer', 'counsel', 'legal', 'statute', 'law',
    'regulation', 'compliance', 'liability', 'damages', 'compensation', 'injunction',
    'subpoena', 'deposition', 'testimony', 'evidence', 'jurisdiction', 'precedent',
    'constitutional', 'federal', 'state', 'municipal', 'criminal', 'civil', 'contract',
    'tort', 'negligence', 'malpractice', 'fraud', 'breach', 'violation', 'penalty',
    'fine', 'sanction', 'appeal', 'motion', 'hearing', 'ruling', 'decision'
]

BRAND_COMPANIES = [
    'apple', 'microsoft', 'google', 'amazon', 'facebook', 'meta', 'tesla', 'netflix',
    'uber', 'airbnb', 'spotify', 'twitter', 'linkedin', 'instagram', 'youtube',
    'tiktok', 'snapchat', 'whatsapp', 'zoom', 'slack', 'discord', 'reddit',
    'paypal', 'stripe', 'square', 'shopify', 'salesforce', 'adobe', 'oracle',
    'ibm', 'intel', 'nvidia', 'amd', 'cisco', 'dell', 'hp', 'lenovo'
]

LAWSUIT_PATTERNS = [
    r'\b(class action|class-action)\b',
    r'\b(settlement|settled)\b',
    r'\b(damages|compensation)\b',
    r'\b(breach of contract|contract breach)\b',
    r'\b(negligence|negligent)\b',
    r'\b(malpractice|professional negligence)\b',
    r'\b(fraud|fraudulent)\b',
    r'\b(liability|liable)\b'
]

LEGAL_REFERENCE_PATTERNS = [
    r'\b(see|refer to|according to|per)\s+[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+\b',  # Case citations
    r'\b\d+\s+[A-Z][a-z]+\s+\d+\b',  # Statute references
    r'\b(CFR|U\.S\.C\.|F\.R\.|F\.3d|F\.2d)\b',  # Legal citation formats
    r'\b(section|sec\.|§)\s+\d+',  # Section references
    r'\b(article|art\.)\s+\d+',  # Article references
    r'\b(paragraph|para\.|¶)\s+\d+'  # Paragraph references
]

def extract_legal_brand_features(text: str) -> dict:
    """
    Extract legal and brand recognition features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_legal_brand_features()
    
    text_lower = text.lower()
    features = {}
    
    # Legal term detection
    legal_count = 0
    for term in LEGAL_TERMS:
        if term in text_lower:
            legal_count += 1
    
    features['legal_terms_count'] = legal_count
    features['legal_terms_density'] = legal_count / len(text.split()) if text.split() else 0
    
    # Brand/company detection
    brand_count = 0
    for brand in BRAND_COMPANIES:
        if brand in text_lower:
            brand_count += 1
    
    features['brand_mentions_count'] = brand_count
    features['brand_mentions_density'] = brand_count / len(text.split()) if text.split() else 0
    
    # Lawsuit pattern detection
    lawsuit_patterns = 0
    for pattern in LAWSUIT_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        lawsuit_patterns += len(matches)
    
    features['lawsuit_patterns_count'] = lawsuit_patterns
    features['has_lawsuit_patterns'] = 1 if lawsuit_patterns > 0 else 0
    
    # Legal reference detection
    legal_refs = 0
    for pattern in LEGAL_REFERENCE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        legal_refs += len(matches)
    
    features['legal_references_count'] = legal_refs
    features['has_legal_references'] = 1 if legal_refs > 0 else 0
    
    # Legal advice indicators
    legal_advice_patterns = [
        r'\b(you should|you must|you need to|i recommend|i suggest)\s+(consult|hire|get|seek)\s+(a\s+)?(lawyer|attorney|legal)\b',
        r'\b(legal advice|legal opinion|legal counsel)\b',
        r'\b(should consult|recommend consulting|seek legal)\b',
        r'\b(get a lawyer|hire an attorney|contact a lawyer)\b'
    ]
    
    advice_count = 0
    for pattern in legal_advice_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        advice_count += len(matches)
    
    features['legal_advice_indicators'] = advice_count
    features['has_legal_advice'] = 1 if advice_count > 0 else 0
    
    return features

def get_empty_legal_brand_features() -> dict:
    """Return empty legal/brand features"""
    return {
        'legal_terms_count': 0,
        'legal_terms_density': 0.0,
        'brand_mentions_count': 0,
        'brand_mentions_density': 0.0,
        'lawsuit_patterns_count': 0,
        'has_lawsuit_patterns': 0,
        'legal_references_count': 0,
        'has_legal_references': 0,
        'legal_advice_indicators': 0,
        'has_legal_advice': 0
    }

def extract_sentiment_features(text: str) -> dict:
    """
    Extract sentiment analysis features using simple pattern matching
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_sentiment_features()
    
    text_lower = text.lower()
    features = {}
    
    # Positive sentiment indicators
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'brilliant',
        'perfect', 'outstanding', 'superb', 'marvelous', 'terrific', 'fabulous', 'incredible',
        'love', 'like', 'enjoy', 'appreciate', 'admire', 'respect', 'praise', 'commend',
        'helpful', 'useful', 'beneficial', 'valuable', 'worthwhile', 'effective', 'successful'
    ]
    
    # Negative sentiment indicators
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike', 'angry',
        'furious', 'outraged', 'disappointed', 'frustrated', 'annoyed', 'upset', 'sad',
        'depressed', 'worried', 'concerned', 'scared', 'afraid', 'fearful', 'anxious',
        'useless', 'worthless', 'pointless', 'stupid', 'idiotic', 'ridiculous', 'absurd'
    ]
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text.split())
    
    features['positive_sentiment_count'] = positive_count
    features['negative_sentiment_count'] = negative_count
    features['positive_sentiment_ratio'] = positive_count / total_words if total_words > 0 else 0
    features['negative_sentiment_ratio'] = negative_count / total_words if total_words > 0 else 0
    features['sentiment_polarity'] = (positive_count - negative_count) / total_words if total_words > 0 else 0
    
    # Emotional intensity indicators
    intensity_words = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'really', 'so']
    intensity_count = sum(1 for word in intensity_words if word in text_lower)
    features['emotional_intensity'] = intensity_count / total_words if total_words > 0 else 0
    
    return features

def get_empty_sentiment_features() -> dict:
    """Return empty sentiment features"""
    return {
        'positive_sentiment_count': 0,
        'negative_sentiment_count': 0,
        'positive_sentiment_ratio': 0.0,
        'negative_sentiment_ratio': 0.0,
        'sentiment_polarity': 0.0,
        'emotional_intensity': 0.0
    }

def extract_formality_features(text: str) -> dict:
    """
    Extract formality score features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_formality_features()
    
    text_lower = text.lower()
    features = {}
    
    # Formal language indicators
    formal_words = [
        'therefore', 'however', 'furthermore', 'moreover', 'consequently', 'nevertheless',
        'accordingly', 'subsequently', 'previously', 'initially', 'ultimately', 'specifically',
        'particularly', 'especially', 'specifically', 'namely', 'i.e.', 'e.g.', 'respectively',
        'respectively', 'respectively', 'respectively', 'respectively', 'respectively'
    ]
    
    # Informal language indicators
    informal_words = [
        'yeah', 'yep', 'nope', 'nah', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta',
        'awesome', 'cool', 'sucks', 'dude', 'bro', 'lol', 'omg', 'wtf', 'btw',
        'imo', 'imho', 'tbh', 'fyi', 'irl', 'af', 'tbh', 'ngl', 'fr', 'no cap'
    ]
    
    # Contractions
    contractions = [
        "don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "didn't", "i'm",
        "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've",
        "they've", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd", "you'd",
        "he'd", "she'd", "we'd", "they'd"
    ]
    
    total_words = len(text.split())
    
    # Count formal words
    formal_count = sum(1 for word in formal_words if word in text_lower)
    features['formal_words_count'] = formal_count
    features['formal_words_ratio'] = formal_count / total_words if total_words > 0 else 0
    
    # Count informal words
    informal_count = sum(1 for word in informal_words if word in text_lower)
    features['informal_words_count'] = informal_count
    features['informal_words_ratio'] = informal_count / total_words if total_words > 0 else 0
    
    # Count contractions
    contraction_count = sum(1 for word in contractions if word in text_lower)
    features['contractions_count'] = contraction_count
    features['contractions_ratio'] = contraction_count / total_words if total_words > 0 else 0
    
    # Calculate formality score (higher = more formal)
    features['formality_score'] = (formal_count - informal_count - contraction_count) / total_words if total_words > 0 else 0
    
    return features

def get_empty_formality_features() -> dict:
    """Return empty formality features"""
    return {
        'formal_words_count': 0,
        'informal_words_count': 0,
        'contractions_count': 0,
        'formal_words_ratio': 0.0,
        'informal_words_ratio': 0.0,
        'contractions_ratio': 0.0,
        'formality_score': 0.0
    }

def extract_question_pattern_features(text: str) -> dict:
    """
    Extract question pattern features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_question_pattern_features()
    
    text_lower = text.lower()
    features = {}
    
    # Question words
    question_words = ['what', 'when', 'where', 'why', 'how', 'who', 'which', 'whom', 'whose']
    question_word_count = sum(1 for word in question_words if word in text_lower)
    
    # Question marks
    question_mark_count = text.count('?')
    
    # Rhetorical question patterns
    rhetorical_patterns = [
        r'\b(why would|how could|what makes you think|don\'t you think|isn\'t it)\b',
        r'\b(obviously|clearly|surely|certainly)\s+\w+\?',
        r'\b(do you really|are you serious|come on)\b'
    ]
    
    rhetorical_count = 0
    for pattern in rhetorical_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        rhetorical_count += len(matches)
    
    total_words = len(text.split())
    
    features['question_marks_count'] = question_mark_count
    features['has_questions'] = 1 if question_mark_count > 0 else 0
    features['question_words_count'] = question_word_count
    features['question_words_ratio'] = question_word_count / total_words if total_words > 0 else 0
    features['rhetorical_questions_count'] = rhetorical_count
    features['has_rhetorical_questions'] = 1 if rhetorical_count > 0 else 0
    
    return features

def get_empty_question_pattern_features() -> dict:
    """Return empty question pattern features"""
    return {
        'question_marks_count': 0,
        'has_questions': 0,
        'question_words_count': 0,
        'question_words_ratio': 0.0,
        'rhetorical_questions_count': 0,
        'has_rhetorical_questions': 0
    }

def calculate_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate domain-specific features (legal/brand, sentiment, formality, questions)"""
    print("Calculating domain-specific features...")
    rows = []
    for idx, row in df.iterrows():
        text = row['comment_text']
        feats = {}
        feats.update(extract_legal_brand_features(text))
        feats.update(extract_sentiment_features(text))
        feats.update(extract_formality_features(text))
        feats.update(extract_question_pattern_features(text))
        rows.append(feats)
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    print(f"Added {len(rows[0]) if rows else 0} domain-specific features")
    return df

# --- Specificity Features ---

def extract_specificity_features(text: str) -> dict:
    """
    Extract specificity features to distinguish generic vs highly specific content
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_specificity_features()
    
    features = {}
    
    # Contact information patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    
    email_count = len(re.findall(email_pattern, text))
    phone_count = len(re.findall(phone_pattern, text))
    url_count = len(re.findall(url_pattern, text))
    
    features['email_count'] = email_count
    features['phone_count'] = phone_count
    features['url_count'] = url_count
    features['contact_info_count'] = email_count + phone_count + url_count
    
    # Specific action patterns
    action_patterns = [
        r'\b(call|email|contact|reach out to|get in touch with)\b',
        r'\b(visit|go to|check out|look at)\b',
        r'\b(download|install|sign up|register|subscribe)\b',
        r'\b(buy|purchase|order|get|obtain)\b'
    ]
    
    specific_action_count = 0
    for pattern in action_patterns:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        specific_action_count += len(matches)
    
    features['specific_action_count'] = specific_action_count
    
    # Specific numbers (dates, amounts, quantities)
    number_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
        r'\b\$[\d,]+\.?\d*\b',  # Money amounts
        r'\b\d+\s*(years?|months?|days?|hours?|minutes?)\b',  # Time periods
        r'\b\d+\s*(percent|%)\b'  # Percentages
    ]
    
    specific_number_count = 0
    for pattern in number_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        specific_number_count += len(matches)
    
    features['specific_number_count'] = specific_number_count
    
    # Specific locations
    location_patterns = [
        r'\b(in|at|near|around)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # City/State names
        r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b'  # Addresses
    ]
    
    specific_location_count = 0
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        specific_location_count += len(matches)
    
    features['specific_location_count'] = specific_location_count
    
    # Generic vs specific phrases
    generic_phrases = [
        'something', 'anything', 'everything', 'nothing', 'somewhere', 'anywhere',
        'everywhere', 'nowhere', 'someone', 'anyone', 'everyone', 'no one',
        'somehow', 'anyhow', 'somewhat', 'anyway', 'whatever', 'whenever',
        'wherever', 'whoever', 'whichever', 'however'
    ]
    
    specific_phrases = [
        'exactly', 'precisely', 'specifically', 'particularly', 'especially',
        'specifically', 'namely', 'i.e.', 'e.g.', 'for example', 'for instance',
        'in particular', 'to be specific', 'more specifically'
    ]
    
    generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
    specific_count = sum(1 for phrase in specific_phrases if phrase in text.lower())
    
    features['generic_phrase_count'] = generic_count
    features['specific_phrase_count'] = specific_count
    
    # Calculate densities
    total_words = len(text.split())
    features['contact_info_density'] = features['contact_info_count'] / total_words if total_words > 0 else 0
    features['specific_action_density'] = features['specific_action_count'] / total_words if total_words > 0 else 0
    features['specific_number_density'] = features['specific_number_count'] / total_words if total_words > 0 else 0
    features['specific_location_density'] = features['specific_location_count'] / total_words if total_words > 0 else 0
    features['generic_phrase_density'] = features['generic_phrase_count'] / total_words if total_words > 0 else 0
    features['specific_phrase_density'] = features['specific_phrase_count'] / total_words if total_words > 0 else 0
    
    # Overall specificity score (higher = more specific)
    specificity_score = (
        features['specific_action_density'] + 
        features['specific_number_density'] + 
        features['specific_location_density'] + 
        features['specific_phrase_density'] - 
        features['generic_phrase_density']
    )
    features['overall_specificity_score'] = specificity_score
    
    return features

def get_empty_specificity_features() -> dict:
    """Return empty specificity features"""
    return {
        'email_count': 0, 'phone_count': 0, 'url_count': 0, 'contact_info_count': 0,
        'specific_action_count': 0, 'specific_number_count': 0, 'specific_location_count': 0,
        'generic_phrase_count': 0, 'specific_phrase_count': 0,
        'contact_info_density': 0.0, 'specific_action_density': 0.0, 'specific_number_density': 0.0,
        'specific_location_density': 0.0, 'generic_phrase_density': 0.0, 'specific_phrase_density': 0.0,
        'overall_specificity_score': 0.0
    }

def calculate_specificity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate specificity features"""
    print("Calculating specificity features...")
    rows = []
    for idx, row in df.iterrows():
        text = row['comment_text']
        feats = extract_specificity_features(text)
        rows.append(feats)
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    print(f"Added {len(rows[0]) if rows else 0} specificity features")
    return df

# --- Advanced Text Processing Features ---

def extract_advanced_tfidf_features(text: str, tfidf_models: dict) -> dict:
    """
    Extract advanced TF-IDF features using multiple vectorization strategies
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_advanced_tfidf_features()
    
    features = {}
    
    # Standard TF-IDF features
    if 'standard' in tfidf_models:
        try:
            tfidf_vector = tfidf_models['standard'].transform([text]).toarray()[0]
            features['standard_tfidf_sum'] = np.sum(tfidf_vector)
            features['standard_tfidf_mean'] = np.mean(tfidf_vector)
            features['standard_tfidf_max'] = np.max(tfidf_vector)
            features['standard_tfidf_std'] = np.std(tfidf_vector)
        except Exception:
            features.update({'standard_tfidf_sum': 0.0, 'standard_tfidf_mean': 0.0, 'standard_tfidf_max': 0.0, 'standard_tfidf_std': 0.0})
    
    # Sublinear TF-IDF features
    if 'sublinear' in tfidf_models:
        try:
            tfidf_vector = tfidf_models['sublinear'].transform([text]).toarray()[0]
            features['sublinear_tfidf_sum'] = np.sum(tfidf_vector)
            features['sublinear_tfidf_mean'] = np.mean(tfidf_vector)
            features['sublinear_tfidf_max'] = np.max(tfidf_vector)
            features['sublinear_tfidf_std'] = np.std(tfidf_vector)
        except Exception:
            features.update({'sublinear_tfidf_sum': 0.0, 'sublinear_tfidf_mean': 0.0, 'sublinear_tfidf_max': 0.0, 'sublinear_tfidf_std': 0.0})
    
    # BM25-like features
    if 'bm25' in tfidf_models:
        try:
            tfidf_vector = tfidf_models['bm25'].transform([text]).toarray()[0]
            features['bm25_sum'] = np.sum(tfidf_vector)
            features['bm25_mean'] = np.mean(tfidf_vector)
            features['bm25_max'] = np.max(tfidf_vector)
            features['bm25_std'] = np.std(tfidf_vector)
        except Exception:
            features.update({'bm25_sum': 0.0, 'bm25_mean': 0.0, 'bm25_max': 0.0, 'bm25_std': 0.0})
    
    return features

def get_empty_advanced_tfidf_features() -> dict:
    """Return empty advanced TF-IDF features"""
    return {
        'standard_tfidf_sum': 0.0, 'standard_tfidf_mean': 0.0, 'standard_tfidf_max': 0.0, 'standard_tfidf_std': 0.0,
        'sublinear_tfidf_sum': 0.0, 'sublinear_tfidf_mean': 0.0, 'sublinear_tfidf_max': 0.0, 'sublinear_tfidf_std': 0.0,
        'bm25_sum': 0.0, 'bm25_mean': 0.0, 'bm25_max': 0.0, 'bm25_std': 0.0
    }

def extract_word_embedding_features(text: str, word_embeddings: dict) -> dict:
    """
    Extract word embedding-like features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_word_embedding_features()
    
    features = {}
    words = text.split()
    
    if not words:
        return get_empty_word_embedding_features()
    
    # Word length statistics
    word_lengths = [len(word) for word in words]
    features['avg_word_length'] = np.mean(word_lengths)
    features['max_word_length'] = np.max(word_lengths)
    features['min_word_length'] = np.min(word_lengths)
    features['word_length_std'] = np.std(word_lengths)
    
    # Character statistics
    features['char_count'] = len(text)
    features['char_count_no_spaces'] = len(text.replace(' ', ''))
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    features['alpha_count'] = sum(1 for c in text if c.isalpha())
    features['special_char_count'] = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    # Word diversity
    unique_words = set(words)
    features['unique_words'] = len(unique_words)
    features['total_words'] = len(words)
    features['word_diversity'] = len(unique_words) / len(words) if words else 0
    
    # Most frequent word
    word_counts = Counter(words)
    most_frequent = word_counts.most_common(1)[0] if word_counts else ('', 0)
    features['most_frequent_word_count'] = most_frequent[1]
    
    # N-gram counts
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
    
    features['bigram_count'] = len(bigrams)
    features['trigram_count'] = len(trigrams)
    features['unique_bigrams'] = len(set(bigrams))
    features['unique_trigrams'] = len(set(trigrams))
    
    return features

def get_empty_word_embedding_features() -> dict:
    """Return empty word embedding features"""
    return {
        'avg_word_length': 0.0, 'max_word_length': 0, 'min_word_length': 0, 'word_length_std': 0.0,
        'char_count': 0, 'char_count_no_spaces': 0, 'digit_count': 0, 'alpha_count': 0, 'special_char_count': 0,
        'unique_words': 0, 'total_words': 0, 'word_diversity': 0.0, 'most_frequent_word_count': 0,
        'bigram_count': 0, 'trigram_count': 0, 'unique_bigrams': 0, 'unique_trigrams': 0
    }

def extract_text_augmentation_features(text: str) -> dict:
    """
    Extract text augmentation features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_text_augmentation_features()
    
    features = {}
    text_lower = text.lower()
    
    # Synonym patterns
    synonym_patterns = [
        r'\b(great|awesome|amazing|fantastic|wonderful)\b',
        r'\b(bad|terrible|awful|horrible|disgusting)\b',
        r'\b(good|nice|fine|okay|alright)\b'
    ]
    
    synonym_count = 0
    for pattern in synonym_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        synonym_count += len(matches)
    
    features['synonym_pattern_count'] = synonym_count
    
    # Translation patterns (common in multilingual content)
    translation_patterns = [
        r'\b(translation|translate|traducción|traducir)\b',
        r'\b(english|spanish|french|german|chinese)\b',
        r'\b(language|idioma|langue|sprache)\b'
    ]
    
    translation_count = 0
    for pattern in translation_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        translation_count += len(matches)
    
    features['translation_pattern_count'] = translation_count
    
    # Repeated words
    words = text.split()
    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    features['repeated_words_count'] = repeated_words
    
    # Frequent words (common words that appear often)
    frequent_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    frequent_count = sum(1 for word in frequent_words if word in text_lower)
    features['frequent_words_count'] = frequent_count
    
    # Sentence structure
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    # Complex words (words with 3+ syllables)
    complex_words = 0
    for word in words:
        if len(word) >= 7:  # Simple heuristic for complex words
            complex_words += 1
    
    features['complex_word_count'] = complex_words
    features['complex_word_ratio'] = complex_words / len(words) if words else 0
    
    return features

def get_empty_text_augmentation_features() -> dict:
    """Return empty text augmentation features"""
    return {
        'synonym_pattern_count': 0, 'translation_pattern_count': 0, 'repeated_words_count': 0,
        'frequent_words_count': 0, 'sentence_count': 0, 'avg_sentence_length': 0.0,
        'complex_word_count': 0, 'complex_word_ratio': 0.0
    }

def extract_bert_sentence_features(text: str) -> dict:
    """
    Extract BERT-like sentence features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_bert_sentence_features()
    
    features = {}
    
    # Sentence segmentation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return get_empty_bert_sentence_features()
    
    # Sentence statistics
    sentence_lengths = [len(s.split()) for s in sentences]
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = np.mean(sentence_lengths)
    features['max_sentence_length'] = np.max(sentence_lengths)
    features['min_sentence_length'] = np.min(sentence_lengths)
    features['sentence_length_std'] = np.std(sentence_lengths)
    
    # Paragraph segmentation (simple heuristic)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if paragraphs:
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = np.mean(paragraph_lengths)
        features['max_paragraph_length'] = np.max(paragraph_lengths)
        features['min_paragraph_length'] = np.min(paragraph_lengths)
        features['paragraph_length_std'] = np.std(paragraph_lengths)
    else:
        features.update({'paragraph_count': 0, 'avg_paragraph_length': 0.0, 'max_paragraph_length': 0, 'min_paragraph_length': 0, 'paragraph_length_std': 0.0})
    
    # Bigram diversity
    words = text.split()
    if len(words) >= 2:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        features['bigram_diversity'] = len(set(bigrams)) / len(bigrams) if bigrams else 0
    else:
        features['bigram_diversity'] = 0.0
    
    return features

def get_empty_bert_sentence_features() -> dict:
    """Return empty BERT sentence features"""
    return {
        'sentence_count': 0, 'avg_sentence_length': 0.0, 'max_sentence_length': 0, 'min_sentence_length': 0, 'sentence_length_std': 0.0,
        'paragraph_count': 0, 'avg_paragraph_length': 0.0, 'max_paragraph_length': 0, 'min_paragraph_length': 0, 'paragraph_length_std': 0.0,
        'bigram_diversity': 0.0
    }

def calculate_advanced_text_processing_features(df: pd.DataFrame, tfidf_models: dict = None) -> pd.DataFrame:
    """Calculate advanced text processing features"""
    print("Calculating advanced text processing features...")
    
    # Create dummy TF-IDF models if not provided
    if tfidf_models is None:
        tfidf_models = {}
        print("Warning: No TF-IDF models provided. Using dummy features.")
    
    rows = []
    for idx, row in df.iterrows():
        text = row['comment_text']
        feats = {}
        feats.update(extract_advanced_tfidf_features(text, tfidf_models))
        feats.update(extract_word_embedding_features(text, {}))
        feats.update(extract_text_augmentation_features(text))
        feats.update(extract_bert_sentence_features(text))
        rows.append(feats)
    
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    print(f"Added {len(rows[0]) if rows else 0} advanced text processing features")
    return df

# --- Feature Selection & Engineering Features ---

def calculate_mutual_information_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """Calculate mutual information features"""
    print("Calculating mutual information features...")
    
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping MI features.")
        return df
    
    # Get numerical columns
    exclude_cols = {'comment_text', 'rule_violation', 'subreddit', 'rule'}
    numerical_cols = [col for col in df.columns if col not in exclude_cols and str(df.dtypes[col]) in ['int64', 'float64']]
    
    if len(numerical_cols) < 2:
        print("Warning: Not enough numerical features for MI calculation.")
        return df
    
    try:
        # Calculate mutual information
        X = df[numerical_cols].fillna(0)
        y = df[target_column].fillna(0)
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create MI-based features
        mi_features = {}
        for i, col in enumerate(numerical_cols):
            mi_features[f'mi_{col}'] = mi_scores[i]
        
        # Add MI features to dataframe
        mi_df = pd.DataFrame([mi_features] * len(df))
        df = pd.concat([df, mi_df], axis=1)
        print(f"Added {len(mi_features)} mutual information features")
        
    except Exception as e:
        print(f"Error calculating mutual information features: {e}")
    
    return df

def calculate_dimensionality_reduction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dimensionality reduction features using PCA"""
    print("Calculating dimensionality reduction features...")
    
    # Get numerical columns
    exclude_cols = {'comment_text', 'rule_violation', 'subreddit', 'rule'}
    numerical_cols = [col for col in df.columns if col not in exclude_cols and str(df.dtypes[col]) in ['int64', 'float64']]
    
    if len(numerical_cols) < 5:
        print("Warning: Not enough numerical features for PCA.")
        return df
    
    try:
        # Apply PCA
        X = df[numerical_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use 5 principal components
        pca = PCA(n_components=5, random_state=42)
        pca_features = pca.fit_transform(X_scaled)
        
        # Add PCA features to dataframe
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_component_{i+1}' for i in range(5)])
        df = pd.concat([df, pca_df], axis=1)
        print(f"Added 5 PCA features")
        
    except Exception as e:
        print(f"Error calculating PCA features: {e}")
    
    return df

def calculate_recursive_feature_elimination_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """Calculate recursive feature elimination features"""
    print("Calculating recursive feature elimination features...")
    
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping RFE features.")
        return df
    
    # Get numerical columns
    exclude_cols = {'comment_text', 'rule_violation', 'subreddit', 'rule'}
    numerical_cols = [col for col in df.columns if col not in exclude_cols and str(df.dtypes[col]) in ['int64', 'float64']]
    
    if len(numerical_cols) < 10:
        print("Warning: Not enough numerical features for RFE.")
        return df
    
    try:
        # Apply RFE
        X = df[numerical_cols].fillna(0)
        y = df[target_column].fillna(0)
        
        # Use Random Forest as base estimator
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=10)
        rfe.fit(X, y)
        
        # Create RFE-based features
        rfe_features = {}
        for i, col in enumerate(numerical_cols):
            rfe_features[f'rfe_{col}'] = 1 if rfe.support_[i] else 0
        
        # Add RFE features to dataframe
        rfe_df = pd.DataFrame([rfe_features] * len(df))
        df = pd.concat([df, rfe_df], axis=1)
        print(f"Added {len(rfe_features)} RFE features")
        
    except Exception as e:
        print(f"Error calculating RFE features: {e}")
    
    return df

def calculate_feature_selection_engineering_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """Calculate comprehensive feature selection and engineering features"""
    print("Calculating feature selection and engineering features...")
    
    # Apply all feature selection methods
    df = calculate_mutual_information_features(df, target_column)
    df = calculate_dimensionality_reduction_features(df)
    df = calculate_recursive_feature_elimination_features(df, target_column)
    
    # Get numerical columns for feature engineering
    exclude_cols = {'comment_text', 'rule_violation', 'subreddit', 'rule'}
    numerical_cols = [col for col in df.columns if col not in exclude_cols and str(df.dtypes[col]) in ['int64', 'float64']]
    
    if len(numerical_cols) < 2:
        print("Warning: Not enough numerical features for feature engineering")
        return df
    
    # Create feature interactions
    interaction_features = {}
    for i, col1 in enumerate(numerical_cols[:10]):  # Limit to first 10 to avoid too many features
        for j, col2 in enumerate(numerical_cols[:10]):
            if i < j:  # Avoid duplicates
                try:
                    # Multiplication interaction
                    interaction_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    # Division interaction (with safety check)
                    safe_div = np.where(df[col2] != 0, df[col1] / df[col2], 0)
                    interaction_features[f'{col1}_div_{col2}'] = safe_div
                except Exception as e:
                    print(f"Warning: Could not create interaction between {col1} and {col2}: {e}")
    
    # Add polynomial features for key columns
    key_cols = ['comment_length', 'exclamation_frequency', 'similarity_to_violation', 'similarity_to_safe']
    for col in key_cols:
        if col in df.columns:
            try:
                interaction_features[f'{col}_squared'] = df[col] ** 2
                interaction_features[f'{col}_cubed'] = df[col] ** 3
                interaction_features[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            except Exception as e:
                print(f"Warning: Could not create polynomial features for {col}: {e}")
    
    # Add the new features to dataframe
    if interaction_features:
        interaction_df = pd.DataFrame(interaction_features)
        df = pd.concat([df, interaction_df], axis=1)
        print(f"Added {len(interaction_features)} feature engineering features")
    else:
        print("No feature engineering features added")
    
    return df

# --- Rule-Specific Features ---

def calculate_rule_specific_features(text: str, rule_patterns: dict, tfidf_model: TfidfVectorizer) -> dict:
    """
    Calculate rule-specific features based on positive and negative examples
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_rule_specific_features()
    
    features = {}
    
    # Rule pattern matching
    for rule_name, patterns in rule_patterns.items():
        pattern_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            pattern_count += len(matches)
        features[f'rule_{rule_name}_pattern_count'] = pattern_count
    
    # TF-IDF similarity to rule patterns
    try:
        text_vector = tfidf_model.transform([text]).toarray()[0]
        for rule_name, rule_vector in rule_patterns.items():
            if 'vector' in rule_vector:
                similarity = cosine_similarity([text_vector], [rule_vector['vector']])[0][0]
                features[f'rule_{rule_name}_similarity'] = similarity
    except Exception:
        # If TF-IDF fails, set similarities to 0
        for rule_name in rule_patterns.keys():
            features[f'rule_{rule_name}_similarity'] = 0.0
    
    return features

def get_empty_rule_specific_features() -> dict:
    """Return empty rule-specific features"""
    return {
        'rule_pos_similarity': 0.0,
        'rule_neg_similarity': 0.0,
        'rule_similarity_diff': 0.0,
        'rule_similarity_ratio': 1.0,
        'rule_diff_alignment': 0.0,
        'rule_pos_consistency': 0.0,
        'rule_neg_consistency': 0.0,
        'rule_consistency_diff': 0.0,
        'rule_pos_var_alignment': 0.0,
        'rule_neg_var_alignment': 0.0,
        'rule_var_alignment_diff': 0.0,
        'rule_violation_score': 0.0
    }

def calculate_rule_specific_comparisons(df: pd.DataFrame, tfidf_model, mean_vectors) -> pd.DataFrame:
    """Calculate rule-specific comparison features"""
    print("Calculating rule-specific comparison features...")
    
    # Create rule-specific features based on available data
    rule_features = {}
    
    # Rule-based similarity features
    if 'similarity_to_violation' in df.columns and 'similarity_to_safe' in df.columns:
        rule_features['rule_similarity_diff'] = df['similarity_to_violation'] - df['similarity_to_safe']
        rule_features['rule_similarity_ratio'] = np.where(
            df['similarity_to_safe'] != 0, 
            df['similarity_to_violation'] / df['similarity_to_safe'], 
            1.0
        )
        rule_features['rule_similarity_sum'] = df['similarity_to_violation'] + df['similarity_to_safe']
        rule_features['rule_similarity_max'] = np.maximum(df['similarity_to_violation'], df['similarity_to_safe'])
        rule_features['rule_similarity_min'] = np.minimum(df['similarity_to_violation'], df['similarity_to_safe'])
    
    # Boundary proximity features
    if 'boundary_proximity_score' in df.columns:
        rule_features['rule_boundary_proximity_abs'] = np.abs(df['boundary_proximity_score'])
        rule_features['rule_boundary_proximity_squared'] = df['boundary_proximity_score'] ** 2
        rule_features['rule_boundary_proximity_sign'] = np.sign(df['boundary_proximity_score'])
    
    # Consistency features
    if 'consistency_deviation' in df.columns:
        rule_features['rule_consistency_abs'] = np.abs(df['consistency_deviation'])
        rule_features['rule_consistency_squared'] = df['consistency_deviation'] ** 2
    
    # Legal advice interaction features
    if 'legal_advice_interaction_feature' in df.columns:
        rule_features['rule_legal_advice_weighted'] = df['legal_advice_interaction_feature'] * df.get('similarity_to_violation', 0)
        rule_features['rule_legal_advice_boundary'] = df['legal_advice_interaction_feature'] * df.get('boundary_proximity_score', 0)
    
    # Promo persuasion features
    if 'promo_persuasion_feature' in df.columns:
        rule_features['rule_promo_weighted'] = df['promo_persuasion_feature'] * df.get('similarity_to_violation', 0)
        rule_features['rule_promo_boundary'] = df['promo_persuasion_feature'] * df.get('boundary_proximity_score', 0)
    
    # Add the new features to dataframe
    if rule_features:
        rule_df = pd.DataFrame(rule_features)
        df = pd.concat([df, rule_df], axis=1)
        print(f"Added {len(rule_features)} rule-specific comparison features")
    else:
        print("No rule-specific comparison features added")
    
    return df

# --- Feature Calculation Functions ---

def calculate_simple_features(df: pd.DataFrame, scaler: RobustScaler = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """Calculates and scales continuous structural features with enhanced preprocessing."""
    
    print("Calculating simple features...")
    
    # Enhanced text preprocessing
    df['comment_text'] = df['comment_text'].apply(_clean_and_normalize_text)
    
    # Basic structural features
    df['comment_length'] = df['comment_text'].str.len()
    df['word_count'] = df['comment_text'].str.split().str.len()
    df['sentence_count'] = df['comment_text'].str.count(r'[.!?]+')
    df['exclamation_frequency'] = df['comment_text'].apply(_get_exclamation_frequency)
    
    # Legal and promotional interaction features
    df['legal_advice_interaction_feature'] = df['comment_text'].apply(_check_legal_advice_interaction)
    df['promo_persuasion_feature'] = df['comment_text'].apply(_calculate_promo_persuasion_feature)
    
    # Select continuous features for scaling
    continuous_features = ['comment_length', 'word_count', 'sentence_count', 'exclamation_frequency']
    
    # Initialize scaler if not provided
    if scaler is None:
        scaler = RobustScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
    else:
        df[continuous_features] = scaler.transform(df[continuous_features])
    
    print(f"Added {len(continuous_features)} simple features")
    return df, scaler

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate interaction features between basic features"""
    print("Calculating interaction features...")
    
    # Create interaction features
    if 'comment_length' in df.columns and 'word_count' in df.columns:
        df['length_word_ratio'] = df['comment_length'] / (df['word_count'] + 1)
    
    if 'exclamation_frequency' in df.columns and 'word_count' in df.columns:
        df['exclamation_word_interaction'] = df['exclamation_frequency'] * df['word_count']
    
    print("Added interaction features")
    return df

def calculate_similarity_features(
    df: pd.DataFrame, 
    tfidf_model: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate similarity features using TF-IDF vectors and mean vectors
    """
    print("Calculating similarity features...")
    
    # Transform text to TF-IDF vectors
    X_tfidf = tfidf_model.transform(df['comment_text']).toarray()
    
    # Calculate similarities to mean vectors
    if 'violation' in mean_vectors and 'safe' in mean_vectors:
        violation_similarities = cosine_similarity(X_tfidf, [mean_vectors['violation']]).flatten()
        safe_similarities = cosine_similarity(X_tfidf, [mean_vectors['safe']]).flatten()
        
        df['similarity_to_violation'] = violation_similarities
        df['similarity_to_safe'] = safe_similarities
        df['similarity_difference'] = violation_similarities - safe_similarities
        df['similarity_ratio'] = np.where(safe_similarities != 0, violation_similarities / safe_similarities, 1.0)
    else:
        # Fallback for test data
        df['similarity_to_violation'] = 0.0
        df['similarity_to_safe'] = 0.0
        df['similarity_difference'] = 0.0
        df['similarity_ratio'] = 1.0
    
    # Calculate boundary proximity using semantic difference vector
    if 'semantic_difference' in mean_vectors:
        semantic_similarities = cosine_similarity(X_tfidf, [mean_vectors['semantic_difference']]).flatten()
        df['boundary_proximity_score'] = semantic_similarities
    else:
        df['boundary_proximity_score'] = 0.0
    
    print("Added similarity features")
    return df

def calculate_consistency_features(
    df: pd.DataFrame, 
    tfidf_model: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """Calculates the consistency deviation feature."""
    
    # Check if the required columns exist
    if 'positive_example_1' not in df.columns or 'positive_example_2' not in df.columns:
        print("Warning: positive_example_1 or positive_example_2 columns not found. Skipping consistency features.")
        df['example_consistency'] = 0.0
        df['consistency_deviation'] = 0.0
        return df
    
    print("Calculating consistency features...")
    
    # Calculate consistency for each row
    consistency_scores = []
    for idx, row in df.iterrows():
        try:
            # Get positive examples
            pos_ex1 = str(row['positive_example_1']) if pd.notna(row['positive_example_1']) else ''
            pos_ex2 = str(row['positive_example_2']) if pd.notna(row['positive_example_2']) else ''
            
            if pos_ex1 and pos_ex2:
                # Transform examples to TF-IDF vectors
                ex1_vector = tfidf_model.transform([pos_ex1]).toarray()[0]
                ex2_vector = tfidf_model.transform([pos_ex2]).toarray()[0]
                
                # Calculate cosine similarity between examples
                similarity = cosine_similarity([ex1_vector], [ex2_vector])[0][0]
                consistency_scores.append(similarity)
            else:
                consistency_scores.append(0.0)
        except Exception as e:
            print(f"Error calculating consistency for row {idx}: {e}")
            consistency_scores.append(0.0)
    
    df['example_consistency'] = consistency_scores
    
    # Calculate consistency deviation (how much the current comment deviates from the consistency)
    if 'semantic_difference' in mean_vectors:
        # Transform current comments to TF-IDF vectors
        X_tfidf = tfidf_model.transform(df['comment_text']).toarray()
        semantic_similarities = cosine_similarity(X_tfidf, [mean_vectors['semantic_difference']]).flatten()
        
        # Calculate deviation from consistency
        df['consistency_deviation'] = semantic_similarities - df['example_consistency']
    else:
        df['consistency_deviation'] = 0.0
    
    # Calculate global consistency statistics
    consistency_mean = np.mean(consistency_scores)
    consistency_std = np.std(consistency_scores)
    
    print(f"Training: Calculated consistency_mean={consistency_mean:.4f}, consistency_std={consistency_std:.4f}")
    
    return df

def calculate_archetype_vector(texts: list, tfidf_model: TfidfVectorizer) -> np.ndarray:
    """
    Calculate archetype vector for a list of texts
    """
    if not texts:
        return np.zeros(tfidf_model.get_feature_names_out().shape[0])
    
    # Transform texts to TF-IDF vectors
    vectors = tfidf_model.transform(texts).toarray()
    
    # Calculate mean vector
    archetype = np.mean(vectors, axis=0)
    
    return archetype

def preprocess_data(
    file_path: str = None, 
    df_to_process: pd.DataFrame = None, 
    tfidf_params: Dict[str, Any] = None,
    tfidf_model: TfidfVectorizer = None,
    mean_vectors: Dict[str, Any] = None,
    scaler: MinMaxScaler = None,
    enable_spacy: bool = False
) -> Tuple[pd.DataFrame, TfidfVectorizer, Dict[str, Any], MinMaxScaler]:
    """
    Orchestrates the loading, feature engineering, and scaling of the data.
    Accepts either a file_path (for initial load) or a DataFrame (for splits).
    """
    
    # 1. Load Data (Fixes UnboundLocalError)
    if df_to_process is not None:
        # Use the provided DataFrame (for train/validation splits)
        df = df_to_process.copy() 
    elif file_path is not None:
        # Load from file (for initial train.csv loading)
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            # Return dummy results and exit safely
            return pd.DataFrame(), TfidfVectorizer(max_features=1), {}, MinMaxScaler()
    else:
        # CRITICAL FIX: Ensure the function exits if no data is provided
        raise ValueError("Must provide either 'file_path' or 'df_to_process'.")

    # --- Enhanced Data Cleaning and Validation ---
    
    # CRITICAL FIX: RENAME 'body' to 'comment_text'
    if 'body' in df.columns:
        df = df.rename(columns={'body': 'comment_text'})
    df.columns = df.columns.str.strip() 
    if 'comment_text' not in df.columns:
        print(f"FATAL ERROR: Text column 'comment_text' not found. Available columns: {list(df.columns)}")
        raise KeyError('comment_text')
    
    # Data quality checks and validation
    print(f"Data validation: {len(df)} rows loaded")
    print(f"Missing values in comment_text: {df['comment_text'].isna().sum()}")
    print(f"Empty comments: {(df['comment_text'].str.len() == 0).sum()}")
    
    # Remove completely empty rows
    initial_rows = len(df)
    df = df[df['comment_text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} empty comment rows")
    
    # Validate label column
    if 'rule_violation' in df.columns:
        label_distribution = df['rule_violation'].value_counts()
        print(f"Label distribution: {dict(label_distribution)}")
        print(f"Class balance: {label_distribution[1] / len(df):.3f} positive class")
    
    # Add missing columns for test data if needed
    if 'subreddit' not in df.columns:
        df['subreddit'] = 'unknown'
        print("Warning: 'subreddit' column not found. Using 'unknown' as default.")
    if 'rule' not in df.columns:
        df['rule'] = 'unknown'
        print("Warning: 'rule' column not found. Using 'unknown' as default.")
    
    # FIX: Explicitly set the label column to the one that actually exists
    LABEL_COLUMNS = ['rule_violation'] 
    
    # 2. Calculate Simple Features and Scale 
    df, scaler = calculate_simple_features(df, scaler)

    # 3. Calculate Interaction Features
    df = calculate_interaction_features(df)
    
    # 4. TF-IDF and Mean Vector Calculation (If processing the TRAINING SET)
    if tfidf_model is None:
        print("Fitting TFIDF and calculating mean vectors for the first time...")
        
        # A. Fit TF-IDF with proven parameters
        tfidf_params = tfidf_params if tfidf_params else {
            'max_features': 5000,           # Back to original size
            'stop_words': 'english',        # Remove common words
            'ngram_range': (1, 2),          # Back to bigrams
            'min_df': 1,                    # Less restrictive
            'max_df': 1.0,                  # Less restrictive
            'sublinear_tf': False,          # Disable sublinear scaling
            'norm': 'l2',                   # Keep L2 normalization
            'smooth_idf': True,             # Keep smooth IDF
            'lowercase': True,              # Convert to lowercase
        }
        tfidf_model = TfidfVectorizer(**tfidf_params)
        X_tfidf = tfidf_model.fit_transform(df['comment_text']).toarray()
        
        # B. Calculate Mean Vectors (only if we have labels)
        if 'rule_violation' in df.columns:
            violation_mask = df['rule_violation'] == 1 
            MEAN_VIOLATION_VECTOR = X_tfidf[violation_mask].mean(axis=0)
            MEAN_SAFE_VECTOR = X_tfidf[~violation_mask].mean(axis=0)
        else:
            # For test data without labels, create dummy vectors
            print("Warning: No rule_violation column found. Creating dummy mean vectors for test data.")
            MEAN_VIOLATION_VECTOR = np.zeros(X_tfidf.shape[1])
            MEAN_SAFE_VECTOR = np.zeros(X_tfidf.shape[1])
        
        # C. Calculate Semantic Difference Vector (Boundary Proximity Feature)
        # Extract positive and negative example texts (only if they exist)
        if all(col in df.columns for col in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']):
            pos_ex1_texts = df['positive_example_1'].astype(str).fillna('').tolist()
            pos_ex2_texts = df['positive_example_2'].astype(str).fillna('').tolist()
            neg_ex1_texts = df['negative_example_1'].astype(str).fillna('').tolist()
            neg_ex2_texts = df['negative_example_2'].astype(str).fillna('').tolist()
            
            # Calculate archetype vectors for positive examples
            ARCHETYPE_VECTOR_1 = calculate_archetype_vector(pos_ex1_texts, tfidf_model)
            ARCHETYPE_VECTOR_2 = calculate_archetype_vector(pos_ex2_texts, tfidf_model)
            
            # Calculate median positive vector (average of the two positive archetypes)
            MEDIAN_POSITIVE_VECTOR = (ARCHETYPE_VECTOR_1 + ARCHETYPE_VECTOR_2) / 2
            
            # Calculate archetype vectors for negative examples
            ARCHETYPE_NEG_VECTOR_1 = calculate_archetype_vector(neg_ex1_texts, tfidf_model)
            ARCHETYPE_NEG_VECTOR_2 = calculate_archetype_vector(neg_ex2_texts, tfidf_model)
            
            # Calculate median negative vector (average of the two negative archetypes)
            MEDIAN_NEGATIVE_VECTOR = (ARCHETYPE_NEG_VECTOR_1 + ARCHETYPE_NEG_VECTOR_2) / 2
            
            # CRITICAL STEP: The Semantic Difference Vector
            SEMANTIC_DIFFERENCE_VECTOR = MEDIAN_POSITIVE_VECTOR - MEDIAN_NEGATIVE_VECTOR
        else:
            # For test data without example columns, create dummy semantic difference vector
            print("Warning: Example columns not found. Creating dummy semantic difference vector for test data.")
            SEMANTIC_DIFFERENCE_VECTOR = np.zeros(X_tfidf.shape[1])
        
        mean_vectors = {
            'violation': MEAN_VIOLATION_VECTOR, 
            'safe': MEAN_SAFE_VECTOR,
            'semantic_difference': SEMANTIC_DIFFERENCE_VECTOR
        }
        
        print(f"Training: Calculated semantic difference vector with shape {SEMANTIC_DIFFERENCE_VECTOR.shape}")
        
    # 5. Calculate Similarity Features (For all datasets)
    # This line now works for both cases because tfidf_model and mean_vectors 
    # are guaranteed to be defined (either passed in or calculated in Step 4)
    df = calculate_similarity_features(df, tfidf_model, mean_vectors)
    
    # 6. Calculate Consistency Features (Global Consistency Feature)
    df = calculate_consistency_features(df, tfidf_model, mean_vectors)
    
    # 7. Calculate Context-Aware Stylometric Features (Subreddit-Rule Specific Patterns)
    df = calculate_context_aware_stylometric_features(df)
    
    # 8. Calculate Advanced Text Features (POS, Dependency, Readability, Lexical Diversity)
    df = calculate_advanced_text_features(df, enable_spacy=enable_spacy)
    
    # 9. Calculate Domain-Specific Features (Legal/Brand Recognition, Sentiment, Formality, Questions)
    df = calculate_domain_specific_features(df)
    
    # 10. Calculate Specificity Features (Generic vs Highly Specific Content)
    df = calculate_specificity_features(df)
    
    # 11. Calculate Advanced Text Processing Features (Word2Vec, BERT, TF-IDF variants, Text Augmentation)
    df = calculate_advanced_text_processing_features(df)
    
    # 12. Calculate Feature Selection & Engineering Features (MI, RFE, Interactions, PCA)
    df = calculate_feature_selection_engineering_features(df)
    
    # 13. Calculate Rule-Specific Comparison Features (Positive vs Negative Examples per Rule)
    df = calculate_rule_specific_comparisons(df, tfidf_model, mean_vectors)
    
    # 14. Final Column Selection (Fixes df_final not defined)
    base_columns = ['comment_text'] + [
        'exclamation_frequency', 
        'legal_advice_interaction_feature', 'promo_persuasion_feature', 
        'similarity_to_violation', 'similarity_to_safe', 'consistency_deviation', 'boundary_proximity_score',
        # Context-aware stylometric features (30 features: 10 base features × 3 comparison types)
        'exclamation_ratio_violation_vs_safe_diff', 'exclamation_ratio_violation_vs_safe_ratio', 'exclamation_ratio_violation_zscore',
        'question_ratio_violation_vs_safe_diff', 'question_ratio_violation_vs_safe_ratio', 'question_ratio_violation_zscore',
        'period_ratio_violation_vs_safe_diff', 'period_ratio_violation_vs_safe_ratio', 'period_ratio_violation_zscore',
        'uppercase_ratio_violation_vs_safe_diff', 'uppercase_ratio_violation_vs_safe_ratio', 'uppercase_ratio_violation_zscore',
        'title_case_ratio_violation_vs_safe_diff', 'title_case_ratio_violation_vs_safe_ratio', 'title_case_ratio_violation_zscore',
        'short_word_ratio_violation_vs_safe_diff', 'short_word_ratio_violation_vs_safe_ratio', 'short_word_ratio_violation_zscore',
        'long_word_ratio_violation_vs_safe_diff', 'long_word_ratio_violation_vs_safe_ratio', 'long_word_ratio_violation_zscore',
        'avg_sentence_length_violation_vs_safe_diff', 'avg_sentence_length_violation_vs_safe_ratio', 'avg_sentence_length_violation_zscore',
        'punctuation_density_violation_vs_safe_diff', 'punctuation_density_violation_vs_safe_ratio', 'punctuation_density_violation_zscore',
        'capitalization_ratio_violation_vs_safe_diff', 'capitalization_ratio_violation_vs_safe_ratio', 'capitalization_ratio_violation_zscore',
        # Advanced text features (filtered for high discrimination)
        # High-value POS features (8 features)
        'pos_adj_ratio', 'pos_adv_ratio', 'pos_aux_ratio', 'pos_conj_ratio', 'pos_intj_ratio', 'pos_pron_ratio', 'pos_propn_ratio', 'pos_verb_ratio',
        # Dependency features (4 features)
        'has_imperative', 'has_conditional', 'has_negation', 'has_auxiliary',
        # Readability features (4 features)
        'flesch_kincaid', 'gunning_fog', 'flesch_reading_ease', 'smog_index',
        # Lexical diversity features (4 features)
        'type_token_ratio', 'lexical_diversity', 'vocabulary_richness', 'most_common_word_ratio',
        # Domain-specific features (24 features)
        # Legal/Brand recognition features (10 features)
        'legal_terms_count', 'legal_terms_density', 'brand_mentions_count', 'brand_mentions_density',
        'lawsuit_patterns_count', 'has_lawsuit_patterns', 'legal_references_count', 'has_legal_references',
        'legal_advice_indicators', 'has_legal_advice',
        # Sentiment features (6 features)
        'positive_sentiment_count', 'negative_sentiment_count', 'positive_sentiment_ratio', 'negative_sentiment_ratio',
        'sentiment_polarity', 'emotional_intensity',
        # Formality features (7 features)
        'formal_words_count', 'informal_words_count', 'contractions_count', 'formal_words_ratio',
        'informal_words_ratio', 'contractions_ratio', 'formality_score',
        # Question pattern features (6 features)
        'question_marks_count', 'has_questions', 'question_words_count', 'question_words_ratio',
        'rhetorical_questions_count', 'has_rhetorical_questions',
        # Specificity features (17 features)
        'email_count', 'phone_count', 'url_count', 'contact_info_count', 'specific_action_count',
        'specific_number_count', 'specific_location_count', 'generic_phrase_count', 'specific_phrase_count',
        'contact_info_density', 'specific_action_density', 'specific_number_density', 'specific_location_density',
        'generic_phrase_density', 'specific_phrase_density', 'overall_specificity_score',
        # Advanced text processing features (48 features)
        # Advanced TF-IDF features (12 features)
        'standard_tfidf_sum', 'standard_tfidf_mean', 'standard_tfidf_max', 'standard_tfidf_std',
        'sublinear_tfidf_sum', 'sublinear_tfidf_mean', 'sublinear_tfidf_max', 'sublinear_tfidf_std',
        'bm25_sum', 'bm25_mean', 'bm25_max', 'bm25_std',
        # Word embedding features (18 features)
        'avg_word_length', 'max_word_length', 'min_word_length', 'word_length_std',
        'char_count', 'char_count_no_spaces', 'digit_count', 'alpha_count', 'special_char_count',
        'unique_words', 'total_words', 'word_diversity', 'most_frequent_word_count',
        'bigram_count', 'trigram_count', 'unique_bigrams', 'unique_trigrams',
        # Text augmentation features (8 features)
        'synonym_pattern_count', 'translation_pattern_count', 'repeated_words_count', 'frequent_words_count',
        'sentence_count', 'avg_sentence_length', 'complex_word_count', 'complex_word_ratio',
        # BERT sentence features (11 features)
        'sentence_count', 'avg_sentence_length', 'max_sentence_length', 'min_sentence_length', 'sentence_length_std',
        'paragraph_count', 'avg_paragraph_length', 'max_paragraph_length', 'min_paragraph_length', 'paragraph_length_std',
        'bigram_diversity',
        # Rule-specific comparison features (12 features)
        'rule_pos_similarity', 'rule_neg_similarity', 'rule_similarity_diff', 'rule_similarity_ratio',
        'rule_diff_alignment', 'rule_pos_consistency', 'rule_neg_consistency', 'rule_consistency_diff',
        'rule_pos_var_alignment', 'rule_neg_var_alignment', 'rule_var_alignment_diff', 'rule_violation_score'
        # Note: Feature selection & engineering features are dynamically generated and added automatically
    ]
    
    # Add label columns only if they exist (for test data without labels)
    columns_to_keep = base_columns.copy()
    for label_col in LABEL_COLUMNS:
        if label_col in df.columns:
            columns_to_keep.append(label_col)

    # CRITICAL FIX: Only keep columns that actually exist in the dataframe
    available_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: {len(missing_columns)} expected columns not found: {missing_columns[:10]}...")
        print(f"Using {len(available_columns)} available columns instead of {len(columns_to_keep)} expected columns")
    
    # CRITICAL FIX: Define df_final before returning it
    df_final = df[available_columns] 

    print(f"Preprocessing complete. Final features: {list(df_final.columns)}")
    
    return df_final, tfidf_model, mean_vectors, scaler

# --------------------------------------------------------------------
# END: preprocess.py inline
# --------------------------------------------------------------------

# ===========================
# TOKENIZER (offline, simple)
# ===========================
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab = {'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3}
        self.vocab_size = vocab_size
        self.pad_token_id=0; self.unk_token_id=1; self.cls_token_id=2; self.sep_token_id=3
        common = ['the','be','to','of','and','a','in','that','have','i','it','for','not','on','with','he','as','you','do','at']
        for i,w in enumerate(common,4): self.vocab[w]=i
    def tokenize(self, text): return str(text).lower().split()
    def convert_tokens_to_ids(self, tokens): return [self.vocab.get(t,1) for t in tokens]
    def __call__(self, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'):
        tokens = ['[CLS]'] + self.tokenize(text) + ['[SEP]']
        ids = self.convert_tokens_to_ids(tokens)
        if truncation and len(ids)>max_length: ids = ids[:max_length-1]+[self.sep_token_id]
        mask = [1]*len(ids)
        if padding=='max_length' and len(ids)<max_length:
            pad = max_length-len(ids)
            ids+= [self.pad_token_id]*pad
            mask+= [0]*pad
        return {'input_ids':torch.tensor([ids]), 'attention_mask':torch.tensor([mask])}

# ===========================
# DATASET (passes all numerics)
# ===========================
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df['comment_text'].astype(str).values
        self.tokenizer = tokenizer
        self.max_length = max_length
        exclude={'comment_text','rule_violation','subreddit','rule'}
        num_cols=[c for c in df.columns if c not in exclude and str(df.dtypes[c]) in ['int64','float64']]
        num_arr = df[num_cols].values if num_cols else np.zeros((len(df),0),dtype=float)
        num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.numerical = torch.tensor(num_arr, dtype=torch.float32)
        if 'rule_violation' in df.columns:
            y = np.nan_to_num(df['rule_violation'].values, nan=0.0)
            self.labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.labels = torch.zeros((len(df),1), dtype=torch.float32)
        print(f"Dataset: {len(self.texts)} samples, numeric_dim={self.numerical.shape[1]}")

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'numerical_features': self.numerical[idx],
            'labels': self.labels[idx]
        }

# ==========================================
# MODEL — fixed 409-dim classifier input
# ==========================================
class CustomTransformerModel(nn.Module):
    def __init__(self, num_rules: int, vocab_size: int = 50000):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.text_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.text_attention = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(409, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_rules)
        )
        self._init_weights()
        print("Model ready: classifier input fixed at 409 dims")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, input_ids, attention_mask, numerical_features):
        emb = self.text_embedding(input_ids)
        emb = emb * attention_mask.unsqueeze(-1).float()
        lstm_out, _ = self.text_lstm(emb)
        attn = torch.softmax(self.text_attention(lstm_out), dim=1)
        text_feat = (lstm_out * attn).sum(dim=1)   # [B, 256]
        text_feat = self.dropout(text_feat)
        cur = torch.cat((text_feat, numerical_features.float()), dim=1)
        if cur.shape[1] < 409:
            pad = torch.zeros(cur.shape[0], 409-cur.shape[1], device=cur.device)
            combined = torch.cat([cur, pad], dim=1)
        elif cur.shape[1] > 409:
            combined = cur[:, :409]
        else:
            combined = cur
        return self.classifier(combined)

# ===========================
# MAIN (Kaggle submission)
# ===========================
if __name__ == '__main__':
    print("=== LOADING AND PREPROCESSING TEST DATA ===")
    
    # Try multiple possible paths for the test data
    test_paths = [
        '/kaggle/input/jigsaw-agile-community-rules/test.csv',
        'test.csv',
        './test.csv',
        '../test.csv'
    ]
    
    test_df = None
    for path in test_paths:
        try:
            test_df = pd.read_csv(path)
            print(f"✅ Test data loaded from {path}: {test_df.shape}")
            break
        except Exception as e:
            print(f"❌ Could not load test data from {path}: {e}")
            continue
    
    if test_df is None:
        print("❌ Could not load test data from any path. Exiting.")
        exit(1)

    # Load pre-trained components first
    print("Loading pre-trained components...")
    
    # Try multiple possible paths for the training components
    possible_paths = [
        '/kaggle/input/reddit_model/pytorch/default/1/training_components.pth',
        'training_components.pth',
        './training_components.pth',
        '../training_components.pth'
    ]
    
    components_loaded = False
    for path in possible_paths:
        try:
            components = torch.load(path, map_location=torch.device('cpu'))
            print(f"✅ Training components loaded successfully from {path}!")
            
            # Extract components
            tfidf_model = components.get('tfidf_model')
            mean_vectors = components.get('mean_vectors', {})
            scaler = components.get('scaler')
            
            # Process test data with pre-trained components
            test_df_processed, _, _, _ = preprocess_data(
        df_to_process=test_df,
                tfidf_model=tfidf_model,
                mean_vectors=mean_vectors,
                scaler=scaler,
        enable_spacy=False
    )
            components_loaded = True
            break
            
        except Exception as e:
            print(f"❌ Could not load from {path}: {e}")
            continue
    
    if not components_loaded:
        print("❌ Could not load training components from any path.")
        print("⚠️  WARNING: This will cause poor performance!")
        print("   The model will be fitted on test data instead of training data.")
        print("   This is why your score is low (0.619 instead of 0.93).")
        print("   To fix this, you need to run the training script first to create training_components.pth")
        print("")
        
        # Try to create components from training data if available
        print("🔧 Attempting to create components from training data...")
        print("   This should fix the low score issue by using proper preprocessing...")
        
        # Try multiple possible paths for the training data (same pattern as test data)
        # Check if we're in Kaggle environment
        import os
        is_kaggle = os.path.exists('/kaggle/input')
        
        if is_kaggle:
            print("🔍 Detected Kaggle environment")
            train_paths = [
                '/kaggle/input/jigsaw-agile-community-rules/train.csv',
                '/kaggle/input/jigsaw-agile-community-rules/data/train.csv',
                'train.csv',
                'data/train.csv'
            ]
            # List available files in Kaggle input
            try:
                import os
                if os.path.exists('/kaggle/input'):
                    print("📁 Available Kaggle input directories:")
                    for item in os.listdir('/kaggle/input'):
                        print(f"   - {item}")
            except Exception:
                pass
        else:
            print("🔍 Detected local environment")
            train_paths = [
                '../data/train.csv',
                'data/train.csv',
                './data/train.csv',
                'train.csv',
                '/Users/mythilygurunathan/Documents/GitHub/jigsaw-community-rules/data/train.csv'
            ]
            # List available files in current directory
            try:
                import os
                print("📁 Available files in current directory:")
                for item in os.listdir('.'):
                    if item.endswith('.csv'):
                        print(f"   - {item}")
            except Exception:
                pass
        
        train_df = None
        for path in train_paths:
            try:
                train_df = pd.read_csv(path)
                print(f"✅ Found training data at {path}: {train_df.shape}")
                if 'rule_violation' in train_df.columns:
                    print(f"   Rule violations: {train_df['rule_violation'].sum()}")
                break
        except Exception as e:
                print(f"❌ Could not load training data from {path}: {e}")
                continue
        
        if train_df is not None and 'rule_violation' in train_df.columns:
            print("🔄 Creating components from training data...")
            print("   This will take a moment as it processes the full training dataset...")
            # Process training data to get components
            train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
                df_to_process=train_df,
                enable_spacy=False
            )
            print("✅ Components created from training data!")
            print("   Now processing test data with proper components...")
            # Now process test data with the proper components
            test_df_processed, _, _, _ = preprocess_data(
                df_to_process=test_df,
                tfidf_model=tfidf_model,
                mean_vectors=mean_vectors,
                scaler=scaler,
                enable_spacy=False
            )
            print("✅ Test data processed with proper components!")
        else:
            print("❌ No training data found. Using test data (poor results expected)...")
            print("   This will cause the low score issue to persist...")
            test_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
                df_to_process=test_df,
                enable_spacy=False
            )

    print(f"Preprocessed shape: {test_df_processed.shape}")
    
    # Debug: Check feature consistency
    print(f"Number of features: {test_df_processed.shape[1]}")
    print(f"Feature columns: {list(test_df_processed.columns)[:10]}...")  # Show first 10 columns
    
    # Check for key features
    key_features = ['similarity_to_violation', 'similarity_to_safe', 'boundary_proximity_score']
    for feat in key_features:
        if feat in test_df_processed.columns:
            print(f"✅ {feat}: {test_df_processed[feat].mean():.4f} ± {test_df_processed[feat].std():.4f}")
        else:
            print(f"❌ Missing key feature: {feat}")

    tokenizer = SimpleTokenizer(vocab_size=50000)
    model = CustomTransformerModel(num_rules=1, vocab_size=50000)

    test_dataset = CustomDataset(test_df_processed, tokenizer, max_length=256)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Attempting to load trained model weights...")
    
    # Try multiple possible paths for the model weights
    model_paths = [
        '/kaggle/input/reddit_model/pytorch/default/1/best_model.pth',
        'best_model.pth',
        './best_model.pth',
        '../best_model.pth'
    ]
    
    model_loaded = False
    for path in model_paths:
        try:
            state = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state, strict=True)
            print(f"✅ Model weights loaded successfully from {path}!")
            model_loaded = True
            break
        except Exception as e:
            print(f"❌ Could not load model from {path}: {e}")
            continue
    
    if not model_loaded:
        print("❌ Could not load model weights from any path.")
        print("Proceeding with randomly initialized weights.")
    
    model.eval()
    preds=[]
    print("Making predictions...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            logits = model(batch['input_ids'], batch['attention_mask'], batch['numerical_features'])
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.flatten())
            
            # Debug: Print first batch details
            if i == 0:
                print(f"Debug: First batch logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                print(f"Debug: First batch probabilities range: {probs.min():.4f} to {probs.max():.4f}")
                print(f"Debug: First batch probabilities: {probs.flatten()}")
                
                # Check if predictions are too extreme (all 0s or all 1s)
                if probs.min() == probs.max():
                    print("⚠️ WARNING: All predictions are identical! This suggests a model issue.")
                elif probs.min() < 0.1 and probs.max() > 0.9:
                    print("✅ Good: Predictions show good range and discrimination")
                else:
                    print(f"⚠️ WARNING: Predictions might be too conservative (range: {probs.min():.3f}-{probs.max():.3f})")

    submission = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': preds})
    submission.to_csv('submission.csv', index=False)
    
    # Final diagnostics
    print(f"Submission saved: {len(submission)} rows; range {min(preds):.4f}-{max(preds):.4f}")
    print(f"Prediction distribution: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}")
    print(f"Predictions < 0.1: {sum(1 for p in preds if p < 0.1)}")
    print(f"Predictions > 0.9: {sum(1 for p in preds if p > 0.9)}")
    print(f"Predictions in [0.4, 0.6]: {sum(1 for p in preds if 0.4 <= p <= 0.6)}")
    
    print("Preview:\n", submission.head(10))
