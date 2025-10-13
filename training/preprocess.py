import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, normalize, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity

# THIS LINE IS CRITICAL:
from sklearn.feature_extraction.text import TfidfVectorizer 

from scipy.spatial.distance import cosine
import numpy as np
from typing import Tuple, Dict, Any
import spacy
import textstat
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


LEXICAL_CUES = r'\b(you should|you must|i suggest|my advice|best way is to)\b'
SEMANTIC_KEYWORDS = r'\b(sue|lawyer|court|filing|testimony|statute|jurisdiction|legal advice)\b'
PROMO_CUES = r'\b(free|limited|giveaway|discount|click here|watch now|c0mpanyname)\b'
OBFUSCATED_NAMES = r'\b(gamify|c0in|fr3e|cIick|Iink)\b' 


# --- Enhanced Text Processing Functions ---

def _clean_and_normalize_text(text: str) -> str:
    """
    Enhanced text cleaning and normalization for better feature extraction.
    """
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs but keep the fact that URLs were present
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Normalize repeated characters (e.g., "soooooo" -> "sooo")
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    
    # Normalize punctuation
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    
    # Remove excessive punctuation but keep some for sentiment
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def _get_exclamation_frequency(comment: str) -> float:
    """Calculates the density of exclamation marks in the comment."""
    comment_len = len(comment)
    if comment_len == 0:
        return 0.0
    return comment.count('!') / comment_len

def _check_legal_advice_interaction(comment: str) -> int:
    """Checks for the presence of both an advice cue AND a legal keyword."""
    text = comment.lower()
    has_cue = re.search(LEXICAL_CUES, text) is not None
    has_keyword = re.search(SEMANTIC_KEYWORDS, text) is not None
    return 1 if has_cue and has_keyword else 0

def _calculate_promo_persuasion_feature(comment: str) -> int:
    """Checks for the presence of promotional cues OR obfuscated names (Spam/Promo signal)."""
    text = comment.lower()
    promo_count = len(re.findall(PROMO_CUES, text))
    obfuscated_count = len(re.findall(OBFUSCATED_NAMES, text))
    return 1 if promo_count > 0 or obfuscated_count > 0 else 0

def extract_stylometric_features(text: str) -> dict:
    """
    Extract stylometric features that capture writing style patterns.
    These features help distinguish between different types of content.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'exclamation_ratio': 0.0,
            'question_ratio': 0.0,
            'period_ratio': 0.0,
            'uppercase_ratio': 0.0,
            'title_case_ratio': 0.0,
            'short_word_ratio': 0.0,
            'long_word_ratio': 0.0,
            'avg_sentence_length': 0.0,
            'punctuation_density': 0.0,
            'capitalization_ratio': 0.0
        }
    
    features = {}
    
    # 1. Punctuation patterns
    text_len = len(text)
    features['exclamation_ratio'] = text.count('!') / text_len
    features['question_ratio'] = text.count('?') / text_len
    features['period_ratio'] = text.count('.') / text_len
    features['punctuation_density'] = sum(1 for c in text if c in '!?.,;:') / text_len
    
    # 2. Capitalization patterns
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / text_len
    words = text.split()
    if words:
        features['title_case_ratio'] = sum(1 for word in words if word.istitle()) / len(words)
        features['capitalization_ratio'] = sum(1 for word in words if any(c.isupper() for c in word)) / len(words)
    else:
        features['title_case_ratio'] = 0.0
        features['capitalization_ratio'] = 0.0
    
    # 3. Word length patterns
    if words:
        features['short_word_ratio'] = sum(1 for word in words if len(word) <= 3) / len(words)
        features['long_word_ratio'] = sum(1 for word in words if len(word) >= 7) / len(words)
    else:
        features['short_word_ratio'] = 0.0
        features['long_word_ratio'] = 0.0
    
    # 4. Sentence structure
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if sentences:
        features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        features['avg_sentence_length'] = 0.0
    
    return features

def calculate_group_stylometric_features(texts: list) -> dict:
    """
    Calculate stylometric features for a group of texts (positive or negative examples)
    """
    if not texts:
        return get_empty_group_features()
    
    all_features = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            features = extract_stylometric_features(text)
            all_features.append(features)
    
    if not all_features:
        return get_empty_group_features()
    
    # Aggregate features across the group
    group_features = {}
    for feature_name in all_features[0].keys():
        values = [f[feature_name] for f in all_features]
        group_features[f'group_{feature_name}_mean'] = np.mean(values)
        group_features[f'group_{feature_name}_std'] = np.std(values) if len(values) > 1 else 0.0
        group_features[f'group_{feature_name}_max'] = np.max(values)
        group_features[f'group_{feature_name}_min'] = np.min(values)
    
    return group_features

def get_empty_group_features() -> dict:
    """Return empty features when no texts are available"""
    base_features = ['exclamation_ratio', 'question_ratio', 'period_ratio', 'uppercase_ratio', 
                    'title_case_ratio', 'short_word_ratio', 'long_word_ratio', 'avg_sentence_length',
                    'punctuation_density', 'capitalization_ratio']
    
    empty_features = {}
    for feature in base_features:
        empty_features[f'group_{feature}_mean'] = 0.0
        empty_features[f'group_{feature}_std'] = 0.0
        empty_features[f'group_{feature}_max'] = 0.0
        empty_features[f'group_{feature}_min'] = 0.0
    
    return empty_features

def create_comparison_features(positive_features: dict, negative_features: dict) -> dict:
    """
    Create features that compare positive vs negative example patterns
    """
    comparison_features = {}
    
    for feature_name in positive_features.keys():
        if feature_name.startswith('group_') and feature_name.endswith('_mean'):
            base_name = feature_name.replace('group_', '').replace('_mean', '')
            
            pos_mean = positive_features[feature_name]
            neg_mean = negative_features[feature_name]
            
            # Difference between positive and negative patterns
            comparison_features[f'{base_name}_violation_vs_safe_diff'] = pos_mean - neg_mean
            
            # Ratio of positive to negative patterns
            if neg_mean != 0:
                comparison_features[f'{base_name}_violation_vs_safe_ratio'] = pos_mean / neg_mean
            else:
                comparison_features[f'{base_name}_violation_vs_safe_ratio'] = 1.0
            
            # Z-score: how much does this comment deviate from the safe pattern?
            pos_std = positive_features.get(f'group_{base_name}_std', 1.0)
            if pos_std != 0:
                comparison_features[f'{base_name}_violation_zscore'] = (pos_mean - neg_mean) / pos_std
            else:
                comparison_features[f'{base_name}_violation_zscore'] = 0.0
    
    return comparison_features

def calculate_context_aware_stylometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate context-aware stylometric features by comparing positive vs negative examples
    within each subreddit-rule combination.
    """
    print("Calculating context-aware stylometric features...")
    
    # First, calculate stylometric features for all example texts
    all_example_features = {}
    
    # Process all positive and negative examples
    for idx, row in df.iterrows():
        subreddit = row['subreddit']
        rule = row['rule']
        key = f"{subreddit}_{rule}"
        
        if key not in all_example_features:
            all_example_features[key] = {'positive': [], 'negative': []}
        
        # Add positive examples
        for col in ['positive_example_1', 'positive_example_2']:
            if col in row and pd.notna(row[col]):
                all_example_features[key]['positive'].append(str(row[col]))
        
        # Add negative examples
        for col in ['negative_example_1', 'negative_example_2']:
            if col in row and pd.notna(row[col]):
                all_example_features[key]['negative'].append(str(row[col]))
    
    # Calculate group features for each subreddit-rule combination
    group_features_cache = {}
    for key, examples in all_example_features.items():
        positive_features = calculate_group_stylometric_features(examples['positive'])
        negative_features = calculate_group_stylometric_features(examples['negative'])
        group_features_cache[key] = {
            'positive': positive_features,
            'negative': negative_features
        }
    
    # Now calculate comparison features for each row
    all_features = []
    
    for idx, row in df.iterrows():
        subreddit = row['subreddit']
        rule = row['rule']
        key = f"{subreddit}_{rule}"
        
        if key in group_features_cache:
            positive_features = group_features_cache[key]['positive']
            negative_features = group_features_cache[key]['negative']
            comparison_features = create_comparison_features(positive_features, negative_features)
        else:
            # Fallback to generic features if no group data available
            comparison_features = get_generic_comparison_features()
        
        all_features.append(comparison_features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} context-aware stylometric features")
    return df

def get_generic_comparison_features() -> dict:
    """Return generic comparison features when no group data is available"""
    base_features = ['exclamation_ratio', 'question_ratio', 'period_ratio', 'uppercase_ratio', 
                    'title_case_ratio', 'short_word_ratio', 'long_word_ratio', 'avg_sentence_length',
                    'punctuation_density', 'capitalization_ratio']
    
    generic_features = {}
    for feature in base_features:
        generic_features[f'{feature}_violation_vs_safe_diff'] = 0.0
        generic_features[f'{feature}_violation_vs_safe_ratio'] = 1.0
        generic_features[f'{feature}_violation_zscore'] = 0.0
    
    return generic_features

# --- Advanced Text Features ---

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

def extract_readability_features(text: str) -> dict:
    """
    Extract readability metrics using textstat with fallback handling
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_readability_features()
    
    try:
        features = {}
        
        # Flesch-Kincaid Grade Level (try different function names)
        try:
            features['flesch_kincaid'] = textstat.flesch_kincaid_grade(text)
        except AttributeError:
            try:
                features['flesch_kincaid'] = textstat.flesch_kincaid(text)
            except AttributeError:
                features['flesch_kincaid'] = 0.0
        
        # Gunning Fog Index
        try:
            features['gunning_fog'] = textstat.gunning_fog(text)
        except AttributeError:
            features['gunning_fog'] = 0.0
        
        # Flesch Reading Ease
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        except AttributeError:
            features['flesch_reading_ease'] = 0.0
        
        # SMOG Index
        try:
            features['smog_index'] = textstat.smog_index(text)
        except AttributeError:
            features['smog_index'] = 0.0
        
        # Average sentence length
        try:
            features['avg_sentence_length_readability'] = textstat.avg_sentence_length(text)
        except AttributeError:
            features['avg_sentence_length_readability'] = 0.0
        
        # Syllable count per word
        try:
            features['avg_syllables_per_word'] = textstat.avg_syllables_per_word(text)
        except AttributeError:
            features['avg_syllables_per_word'] = 0.0
        
        return features
        
    except Exception as e:
        print(f"Error in readability feature extraction: {e}")
        return get_empty_readability_features()

def get_empty_readability_features() -> dict:
    """Return empty readability features"""
    return {
        'flesch_kincaid': 0.0,
        'gunning_fog': 0.0,
        'flesch_reading_ease': 0.0,
        'smog_index': 0.0,
        'avg_sentence_length_readability': 0.0,
        'avg_syllables_per_word': 0.0
    }

def extract_lexical_diversity_features(text: str) -> dict:
    """
    Extract lexical diversity features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_lexical_features()
    
    try:
        words = text.lower().split()
        if not words:
            return get_empty_lexical_features()
        
        # Type-Token Ratio (unique words / total words)
        unique_words = set(words)
        type_token_ratio = len(unique_words) / len(words)
        
        # Lexical Diversity (using spaCy for lemmatization)
        try:
            # Simple lexical diversity without spaCy dependency
            words = [word.lower() for word in text.split() if word.isalpha()]
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0
        except:
            lexical_diversity = type_token_ratio  # Fallback
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
        
        # Vocabulary richness (unique words per 100 words)
        vocabulary_richness = (len(unique_words) / len(words)) * 100 if len(words) > 0 else 0
        
        # Word frequency distribution
        word_freq = Counter(words)
        most_common_ratio = word_freq.most_common(1)[0][1] / len(words) if len(words) > 0 else 0
        
        return {
            'type_token_ratio': type_token_ratio,
            'lexical_diversity': lexical_diversity,
            'avg_word_length_lexical': avg_word_length,
            'vocabulary_richness': vocabulary_richness,
            'most_common_word_ratio': most_common_ratio
        }
        
    except Exception as e:
        print(f"Error in lexical diversity feature extraction: {e}")
        return get_empty_lexical_features()

def get_empty_lexical_features() -> dict:
    """Return empty lexical diversity features"""
    return {
        'type_token_ratio': 0.0,
        'lexical_diversity': 0.0,
        'avg_word_length_lexical': 0.0,
        'vocabulary_richness': 0.0,
        'most_common_word_ratio': 0.0
    }

def calculate_advanced_text_features(df: pd.DataFrame, enable_spacy: bool = False) -> pd.DataFrame:
    """
    Calculate all advanced text features (POS, dependency, readability, lexical diversity)
    """
    print("Calculating advanced text features...")
    
    # Load spaCy model only if enabled (it's very slow)
    nlp = None
    if enable_spacy:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except OSError:
            print("Warning: spaCy model not found. Using fallback features.")
            nlp = None
    else:
        print("spaCy features disabled for performance. Using fallback features.")
    
    all_features = []
    
    for idx, row in df.iterrows():
        text = row['comment_text']
        
        if idx % 1000 == 0:
            print(f"Processing text {idx}/{len(df)}")
        
        # Combine all advanced features
        features = {}
        
        # POS and dependency features (if spaCy available and enabled)
        if nlp and enable_spacy:
            try:
                features.update(extract_pos_features(text, nlp))
                features.update(extract_dependency_features(text, nlp))
            except Exception as e:
                print(f"Error processing text {idx} with spaCy: {e}")
                features.update(get_empty_pos_features())
                features.update(get_empty_dependency_features())
        else:
            features.update(get_empty_pos_features())
            features.update(get_empty_dependency_features())
        
        # Readability features (fast)
        features.update(extract_readability_features(text))
        
        # Lexical diversity features (fast)
        features.update(extract_lexical_diversity_features(text))
        
        all_features.append(features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} advanced text features")
    return df

# --- Domain-Specific Features ---

# Legal and Brand Recognition Patterns
LEGAL_TERMS = [
    # Legal actions
    'lawsuit', 'litigation', 'sue', 'suing', 'sued', 'plaintiff', 'defendant', 'settlement', 'verdict', 'judgment',
    'court', 'trial', 'hearing', 'deposition', 'testimony', 'evidence', 'witness', 'jury', 'judge', 'attorney',
    'lawyer', 'counsel', 'legal counsel', 'barrister', 'solicitor', 'paralegal', 'legal aid',
    
    # Legal documents and processes
    'filing', 'filed', 'motion', 'brief', 'petition', 'complaint', 'summons', 'subpoena', 'warrant', 'injunction',
    'restraining order', 'cease and desist', 'legal notice', 'contract', 'agreement', 'terms of service',
    'privacy policy', 'disclaimer', 'liability', 'damages', 'compensation', 'restitution',
    
    # Legal concepts
    'jurisdiction', 'statute', 'regulation', 'law', 'legal', 'legally', 'illegal', 'unlawful', 'criminal',
    'civil', 'constitutional', 'federal', 'state', 'municipal', 'precedent', 'case law', 'common law',
    'tort', 'negligence', 'malpractice', 'fraud', 'breach', 'violation', 'infringement', 'copyright',
    'trademark', 'patent', 'intellectual property', 'defamation', 'libel', 'slander', 'harassment',
    'discrimination', 'employment law', 'labor law', 'contract law', 'property law', 'family law',
    'criminal law', 'constitutional law', 'administrative law', 'tax law', 'immigration law',
    
    # Legal advice indicators
    'legal advice', 'legal opinion', 'legal counsel', 'legal representation', 'legal services',
    'should consult', 'recommend consulting', 'seek legal', 'get a lawyer', 'hire an attorney',
    'legal action', 'legal recourse', 'legal remedy', 'legal rights', 'legal obligations'
]

BRAND_COMPANIES = [
    # Major tech companies
    'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'twitter', 'x.com', 'instagram',
    'linkedin', 'youtube', 'tiktok', 'snapchat', 'netflix', 'spotify', 'uber', 'lyft', 'airbnb',
    'tesla', 'spacex', 'openai', 'chatgpt', 'nvidia', 'intel', 'amd', 'ibm', 'oracle', 'salesforce',
    
    # Financial companies
    'bank of america', 'chase', 'wells fargo', 'citibank', 'goldman sachs', 'morgan stanley',
    'jpmorgan', 'berkshire hathaway', 'visa', 'mastercard', 'american express', 'paypal',
    'venmo', 'square', 'stripe', 'coinbase', 'robinhood', 'etrade', 'fidelity', 'vanguard',
    
    # Retail and e-commerce
    'walmart', 'target', 'costco', 'home depot', 'lowes', 'best buy', 'macy\'s', 'nordstrom',
    'ebay', 'etsy', 'shopify', 'alibaba', 'jd.com', 'wish', 'wayfair', 'overstock',
    
    # Media and entertainment
    'disney', 'warner bros', 'universal', 'sony', 'nintendo', 'xbox', 'playstation', 'steam',
    'hulu', 'hbo', 'paramount', 'viacom', 'cbs', 'nbc', 'abc', 'fox', 'cnn', 'cnbc',
    
    # Automotive
    'ford', 'general motors', 'gm', 'chrysler', 'honda', 'toyota', 'nissan', 'bmw', 'mercedes',
    'audi', 'volkswagen', 'volvo', 'hyundai', 'kia', 'subaru', 'mazda', 'lexus', 'infiniti',
    
    # Healthcare and pharmaceuticals
    'pfizer', 'moderna', 'johnson & johnson', 'merck', 'novartis', 'roche', 'bayer', 'gsk',
    'abbott', 'medtronic', 'unitedhealth', 'anthem', 'cigna', 'aetna', 'humana', 'kaiser',
    
    # Food and beverage
    'mcdonalds', 'starbucks', 'coca cola', 'pepsi', 'nestle', 'kraft', 'general mills',
    'kellogg', 'p&g', 'unilever', 'danone', 'mondelez', 'hershey', 'mars', 'ferrero'
]

LAWSUIT_PATTERNS = [
    # Lawsuit indicators
    r'\b(class action|class-action)\b',
    r'\b(mass tort|mass-tort)\b',
    r'\b(collective action|collective-action)\b',
    r'\b(join the lawsuit|join lawsuit|join the class)\b',
    r'\b(file a lawsuit|filing lawsuit|sue for)\b',
    r'\b(settlement fund|settlement money|settlement check)\b',
    r'\b(compensation claim|damages claim|injury claim)\b',
    r'\b(legal action against|suing for|lawsuit against)\b',
    r'\b(attorney fees|legal fees|court costs)\b',
    r'\b(verdict in favor|ruling against|judgment for)\b'
]

LEGAL_REFERENCE_PATTERNS = [
    # Legal citations
    r'\b\d+\s+[A-Z]\.\s*\d+\b',  # Case citations like "123 F.3d 456"
    r'\b[A-Z]+\s+v\.\s+[A-Z]+\b',  # Case names like "Smith v. Jones"
    r'\b\d+\s+U\.S\.C\.\s*\d+\b',  # USC citations
    r'\b\d+\s+C\.F\.R\.\s*\d+\b',  # CFR citations
    r'\b(act of \d{4}|public law \d+)\b',  # Acts and public laws
    r'\b(amendment \d+|first amendment|second amendment)\b',  # Constitutional amendments
    r'\b(section \d+|subsection \d+|paragraph \d+)\b',  # Legal sections
    r'\b(rule \d+|regulation \d+|standard \d+)\b'  # Rules and regulations
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
        'approximately', 'substantially', 'significantly', 'considerably', 'relatively'
    ]
    
    # Informal language indicators
    informal_words = [
        'yeah', 'yep', 'nope', 'nah', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta',
        'awesome', 'cool', 'sucks', 'dude', 'bro', 'man', 'guys', 'folks', 'peeps',
        'lol', 'omg', 'wtf', 'btw', 'fyi', 'imo', 'tbh', 'nvm', 'idk', 'ikr'
    ]
    
    # Contractions
    contractions = [
        "don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "didn't", "i'm",
        "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've",
        "they've", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd", "you'd",
        "he'd", "she'd", "we'd", "they'd"
    ]
    
    # Count formal and informal indicators
    formal_count = sum(1 for word in formal_words if word in text_lower)
    informal_count = sum(1 for word in informal_words if word in text_lower)
    contraction_count = sum(1 for contraction in contractions if contraction in text_lower)
    
    total_words = len(text.split())
    
    features['formal_words_count'] = formal_count
    features['informal_words_count'] = informal_count
    features['contractions_count'] = contraction_count
    features['formal_words_ratio'] = formal_count / total_words if total_words > 0 else 0
    features['informal_words_ratio'] = informal_count / total_words if total_words > 0 else 0
    features['contractions_ratio'] = contraction_count / total_words if total_words > 0 else 0
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
    Extract question pattern detection features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_question_features()
    
    features = {}
    
    # Question mark count
    question_marks = text.count('?')
    features['question_marks_count'] = question_marks
    features['has_questions'] = 1 if question_marks > 0 else 0
    
    # Question words
    question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom']
    question_word_count = 0
    for word in question_words:
        question_word_count += text.lower().count(word)
    
    features['question_words_count'] = question_word_count
    features['question_words_ratio'] = question_word_count / len(text.split()) if text.split() else 0
    
    # Rhetorical question patterns
    rhetorical_patterns = [
        r'\b(why would|why should|why not|how could|how would|what if)\b',
        r'\b(isn\'t it|aren\'t you|don\'t you|wouldn\'t you|shouldn\'t you)\b',
        r'\b(right\?|correct\?|true\?|agree\?|understand\?)\b'
    ]
    
    rhetorical_count = 0
    for pattern in rhetorical_patterns:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        rhetorical_count += len(matches)
    
    features['rhetorical_questions_count'] = rhetorical_count
    features['has_rhetorical_questions'] = 1 if rhetorical_count > 0 else 0
    
    return features

def get_empty_question_features() -> dict:
    """Return empty question features"""
    return {
        'question_marks_count': 0,
        'has_questions': 0,
        'question_words_count': 0,
        'question_words_ratio': 0.0,
        'rhetorical_questions_count': 0,
        'has_rhetorical_questions': 0
    }

def calculate_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all domain-specific features (legal/brand recognition, sentiment, formality, questions)
    """
    print("Calculating domain-specific features...")
    
    all_features = []
    
    for idx, row in df.iterrows():
        text = row['comment_text']
        
        # Combine all domain-specific features
        features = {}
        features.update(extract_legal_brand_features(text))
        features.update(extract_sentiment_features(text))
        features.update(extract_formality_features(text))
        features.update(extract_question_pattern_features(text))
        
        all_features.append(features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} domain-specific features")
    return df

# --- Rule-Specific Comparison Features ---

def extract_specificity_features(text: str) -> dict:
    """
    Extract specificity features that distinguish between generic and highly specific content
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_specificity_features()
    
    text_lower = text.lower()
    features = {}
    
    # 1. Contact Information Specificity
    # Email patterns
    email_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Standard emails
        r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b',  # Emails with spaces
        r'\b[A-Za-z0-9._%+-]+\s*\(\s*at\s*\)\s*[A-Za-z0-9.-]+\s*\(\s*dot\s*\)\s*[A-Z|a-z]{2,}\b'  # Obfuscated emails
    ]
    
    email_count = 0
    for pattern in email_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        email_count += len(matches)
    
    # Phone number patterns
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone numbers
        r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # US phone with parentheses
        r'\b\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone with country code
        r'\b\d{3}\s*\d{3}\s*\d{4}\b'  # US phone with spaces
    ]
    
    phone_count = 0
    for pattern in phone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phone_count += len(matches)
    
    # URL patterns
    url_patterns = [
        r'https?://[^\s]+',  # HTTP/HTTPS URLs
        r'www\.[^\s]+',  # WWW URLs
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'  # Domain names
    ]
    
    url_count = 0
    for pattern in url_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        url_count += len(matches)
    
    features['email_count'] = email_count
    features['phone_count'] = phone_count
    features['url_count'] = url_count
    features['contact_info_count'] = email_count + phone_count + url_count
    
    # 2. Specific Action/Service Patterns
    specific_actions = [
        # Legal services
        r'\b(obtain|get|acquire|secure|procure)\s+(authentic|genuine|real|valid|legal)\s+(identity|id|passport|visa|license|permit|document)\b',
        r'\b(apply|submit|file|register)\s+(for|to)\s+(identity|id|passport|visa|license|permit|document)\b',
        r'\b(contact|call|email|reach)\s+(us|me|them)\s+(for|to)\s+(identity|id|passport|visa|license|permit|document)\b',
        
        # Financial services
        r'\b(transfer|send|receive|deposit|withdraw)\s+(money|cash|funds|bitcoin|crypto)\b',
        r'\b(invest|buy|sell|trade)\s+(stocks|shares|bonds|securities|crypto|bitcoin)\b',
        r'\b(loan|borrow|lend|credit|debt)\s+(money|cash|funds)\b',
        
        # Medical services
        r'\b(prescription|medication|drug|pill|tablet|capsule)\s+(delivery|shipping|mailing)\b',
        r'\b(medical|health|treatment|therapy|surgery)\s+(consultation|appointment|service)\b',
        
        # Educational services
        r'\b(degree|diploma|certificate|qualification)\s+(for|in)\s+(sale|purchase|buy)\b',
        r'\b(essay|paper|assignment|homework)\s+(writing|help|service)\b',
        
        # General services
        r'\b(service|help|assistance|support)\s+(available|offered|provided)\b',
        r'\b(guaranteed|promised|assured)\s+(result|outcome|success)\b'
    ]
    
    specific_action_count = 0
    for pattern in specific_actions:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        specific_action_count += len(matches)
    
    features['specific_action_count'] = specific_action_count
    
    # 3. Specificity Indicators
    # Specific numbers/quantities
    number_patterns = [
        r'\b\d+\s*(dollars?|bucks?|USD|EUR|GBP|CAD|AUD)\b',  # Money amounts
        r'\b\d+\s*(percent|%|per cent)\b',  # Percentages
        r'\b\d+\s*(days?|weeks?|months?|years?)\b',  # Time periods
        r'\b\d+\s*(hours?|minutes?|seconds?)\b',  # Time durations
        r'\b\d+\s*(miles?|kilometers?|km|meters?|feet?|inches?)\b',  # Distances
        r'\b\d+\s*(pounds?|kg|kilograms?|grams?|ounces?)\b'  # Weights
    ]
    
    specific_number_count = 0
    for pattern in number_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        specific_number_count += len(matches)
    
    # Specific locations
    location_patterns = [
        r'\b(in|at|near|around)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # City names
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)\b',  # Street addresses
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(state|province|county|district|region)\b'  # Administrative regions
    ]
    
    specific_location_count = 0
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        specific_location_count += len(matches)
    
    features['specific_number_count'] = specific_number_count
    features['specific_location_count'] = specific_location_count
    
    # 4. Generic vs Specific Language Patterns
    # Generic phrases (safe content)
    generic_phrases = [
        r'\b(here is|here\'s|this is|that is|it is)\s+(the|a|an)\s+(link|url|website|page)\b',
        r'\b(you can|you could|you might|you may)\s+(try|check|look|see|find)\b',
        r'\b(i think|i believe|i guess|i suppose|maybe|perhaps|possibly)\b',
        r'\b(in my opinion|from my experience|as far as i know|to my knowledge)\b',
        r'\b(just|simply|basically|essentially|generally|usually|typically)\b'
    ]
    
    generic_count = 0
    for pattern in generic_phrases:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        generic_count += len(matches)
    
    # Specific phrases (violation content)
    specific_phrases = [
        r'\b(obtain|acquire|secure|procure|get)\s+(authentic|genuine|real|valid|legal)\b',
        r'\b(guaranteed|promised|assured|certified|verified)\s+(result|outcome|success|delivery)\b',
        r'\b(immediately|instantly|right away|asap|urgent|emergency)\b',
        r'\b(confidential|private|secure|safe|protected)\s+(service|process|method|way)\b',
        r'\b(professional|expert|specialist|qualified|licensed)\s+(service|help|assistance)\b'
    ]
    
    specific_phrase_count = 0
    for pattern in specific_phrases:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        specific_phrase_count += len(matches)
    
    features['generic_phrase_count'] = generic_count
    features['specific_phrase_count'] = specific_phrase_count
    
    # 5. Overall Specificity Score
    total_words = len(text.split())
    
    # Calculate specificity ratios
    features['contact_info_density'] = features['contact_info_count'] / total_words if total_words > 0 else 0
    features['specific_action_density'] = features['specific_action_count'] / total_words if total_words > 0 else 0
    features['specific_number_density'] = features['specific_number_count'] / total_words if total_words > 0 else 0
    features['specific_location_density'] = features['specific_location_count'] / total_words if total_words > 0 else 0
    features['generic_phrase_density'] = features['generic_phrase_count'] / total_words if total_words > 0 else 0
    features['specific_phrase_density'] = features['specific_phrase_count'] / total_words if total_words > 0 else 0
    
    # Overall specificity score (higher = more specific/violation-like)
    specificity_score = (
        0.3 * features['contact_info_density'] +
        0.25 * features['specific_action_density'] +
        0.15 * features['specific_number_density'] +
        0.15 * features['specific_location_density'] +
        0.1 * features['specific_phrase_density'] -
        0.05 * features['generic_phrase_density']  # Generic phrases reduce specificity
    )
    features['overall_specificity_score'] = specificity_score
    
    return features

def get_empty_specificity_features() -> dict:
    """Return empty specificity features"""
    return {
        'email_count': 0,
        'phone_count': 0,
        'url_count': 0,
        'contact_info_count': 0,
        'specific_action_count': 0,
        'specific_number_count': 0,
        'specific_location_count': 0,
        'generic_phrase_count': 0,
        'specific_phrase_count': 0,
        'contact_info_density': 0.0,
        'specific_action_density': 0.0,
        'specific_number_density': 0.0,
        'specific_location_density': 0.0,
        'generic_phrase_density': 0.0,
        'specific_phrase_density': 0.0,
        'overall_specificity_score': 0.0
    }

def calculate_specificity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate specificity features for all comments
    """
    print("Calculating specificity features...")
    
    all_features = []
    
    for idx, row in df.iterrows():
        text = row['comment_text']
        features = extract_specificity_features(text)
        all_features.append(features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} specificity features")
    return df

# --- Advanced Text Processing Features ---

def extract_advanced_tfidf_features(text: str, tfidf_models: dict) -> dict:
    """
    Extract advanced TF-IDF features using different variants
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_advanced_tfidf_features()
    
    features = {}
    
    # 1. Standard TF-IDF features
    if 'standard' in tfidf_models:
        try:
            standard_vector = tfidf_models['standard'].transform([text])
            features['standard_tfidf_sum'] = standard_vector.sum()
            features['standard_tfidf_mean'] = standard_vector.mean()
            features['standard_tfidf_max'] = standard_vector.max()
            features['standard_tfidf_std'] = standard_vector.std()
        except:
            features.update(get_empty_advanced_tfidf_features())
    
    # 2. Sublinear TF-IDF features
    if 'sublinear' in tfidf_models:
        try:
            sublinear_vector = tfidf_models['sublinear'].transform([text])
            features['sublinear_tfidf_sum'] = sublinear_vector.sum()
            features['sublinear_tfidf_mean'] = sublinear_vector.mean()
            features['sublinear_tfidf_max'] = sublinear_vector.max()
            features['sublinear_tfidf_std'] = sublinear_vector.std()
        except:
            features.update(get_empty_advanced_tfidf_features())
    
    # 3. BM25-style features
    if 'bm25' in tfidf_models:
        try:
            bm25_vector = tfidf_models['bm25'].transform([text])
            features['bm25_sum'] = bm25_vector.sum()
            features['bm25_mean'] = bm25_vector.mean()
            features['bm25_max'] = bm25_vector.max()
            features['bm25_std'] = bm25_vector.std()
        except:
            features.update(get_empty_advanced_tfidf_features())
    
    return features

def get_empty_advanced_tfidf_features() -> dict:
    """Return empty advanced TF-IDF features"""
    return {
        'standard_tfidf_sum': 0.0,
        'standard_tfidf_mean': 0.0,
        'standard_tfidf_max': 0.0,
        'standard_tfidf_std': 0.0,
        'sublinear_tfidf_sum': 0.0,
        'sublinear_tfidf_mean': 0.0,
        'sublinear_tfidf_max': 0.0,
        'sublinear_tfidf_std': 0.0,
        'bm25_sum': 0.0,
        'bm25_mean': 0.0,
        'bm25_max': 0.0,
        'bm25_std': 0.0
    }

def extract_word_embedding_features(text: str, word_embeddings: dict) -> dict:
    """
    Extract Word2Vec/FastText embedding features
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_word_embedding_features()
    
    features = {}
    
    # Simple word-based features (since we don't have actual Word2Vec models loaded)
    words = text.lower().split()
    
    # 1. Word length statistics
    word_lengths = [len(word) for word in words]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    features['max_word_length'] = np.max(word_lengths) if word_lengths else 0
    features['min_word_length'] = np.min(word_lengths) if word_lengths else 0
    features['word_length_std'] = np.std(word_lengths) if word_lengths else 0
    
    # 2. Character-level features
    features['char_count'] = len(text)
    features['char_count_no_spaces'] = len(text.replace(' ', ''))
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    features['alpha_count'] = sum(1 for c in text if c.isalpha())
    features['special_char_count'] = len(text) - sum(1 for c in text if c.isalnum() or c.isspace())
    
    # 3. Word frequency features
    word_freq = Counter(words)
    features['unique_words'] = len(word_freq)
    features['total_words'] = len(words)
    features['word_diversity'] = features['unique_words'] / features['total_words'] if features['total_words'] > 0 else 0
    features['most_frequent_word_count'] = max(word_freq.values()) if word_freq else 0
    
    # 4. N-gram features (bigrams and trigrams)
    bigrams = [words[i] + '_' + words[i+1] for i in range(len(words)-1)]
    trigrams = [words[i] + '_' + words[i+1] + '_' + words[i+2] for i in range(len(words)-2)]
    
    features['bigram_count'] = len(bigrams)
    features['trigram_count'] = len(trigrams)
    features['unique_bigrams'] = len(set(bigrams))
    features['unique_trigrams'] = len(set(trigrams))
    
    return features

def get_empty_word_embedding_features() -> dict:
    """Return empty word embedding features"""
    return {
        'avg_word_length': 0.0,
        'max_word_length': 0,
        'min_word_length': 0,
        'word_length_std': 0.0,
        'char_count': 0,
        'char_count_no_spaces': 0,
        'digit_count': 0,
        'alpha_count': 0,
        'special_char_count': 0,
        'unique_words': 0,
        'total_words': 0,
        'word_diversity': 0.0,
        'most_frequent_word_count': 0,
        'bigram_count': 0,
        'trigram_count': 0,
        'unique_bigrams': 0,
        'unique_trigrams': 0
    }

def extract_text_augmentation_features(text: str) -> dict:
    """
    Extract text augmentation features (synonym replacement, back translation patterns)
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_text_augmentation_features()
    
    text_lower = text.lower()
    features = {}
    
    # 1. Synonym replacement patterns
    synonym_patterns = [
        # Common synonym pairs
        r'\b(good|great|excellent|amazing|wonderful|fantastic|awesome|brilliant)\b',
        r'\b(bad|terrible|awful|horrible|disgusting|hate|dislike|angry)\b',
        r'\b(big|large|huge|enormous|massive|giant|colossal)\b',
        r'\b(small|little|tiny|miniature|petite|compact|mini)\b',
        r'\b(fast|quick|rapid|swift|speedy|hasty|brisk)\b',
        r'\b(slow|sluggish|leisurely|gradual|delayed|tardy)\b',
        r'\b(beautiful|gorgeous|stunning|lovely|attractive|pretty|handsome)\b',
        r'\b(ugly|hideous|repulsive|unattractive|unsightly|grotesque)\b'
    ]
    
    synonym_count = 0
    for pattern in synonym_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        synonym_count += len(matches)
    
    features['synonym_pattern_count'] = synonym_count
    
    # 2. Back translation patterns (common translation artifacts)
    translation_patterns = [
        r'\b(very much|so much|too much|much more|much less)\b',
        r'\b(in order to|so as to|for the purpose of)\b',
        r'\b(it is important|it is necessary|it is essential)\b',
        r'\b(as well as|in addition to|furthermore|moreover)\b',
        r'\b(on the other hand|however|nevertheless|nonetheless)\b',
        r'\b(in my opinion|from my point of view|as far as i am concerned)\b'
    ]
    
    translation_count = 0
    for pattern in translation_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        translation_count += len(matches)
    
    features['translation_pattern_count'] = translation_count
    
    # 3. Repetition patterns (common in augmented text)
    words = text_lower.split()
    word_freq = Counter(words)
    
    # Count repeated words
    repeated_words = sum(1 for count in word_freq.values() if count > 1)
    features['repeated_words_count'] = repeated_words
    
    # Count words that appear more than twice
    frequent_words = sum(1 for count in word_freq.values() if count > 2)
    features['frequent_words_count'] = frequent_words
    
    # 4. Sentence structure patterns
    sentences = text.split('.')
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
    
    # 5. Language complexity indicators
    complex_words = [word for word in words if len(word) > 6]
    features['complex_word_count'] = len(complex_words)
    features['complex_word_ratio'] = len(complex_words) / len(words) if words else 0
    
    return features

def get_empty_text_augmentation_features() -> dict:
    """Return empty text augmentation features"""
    return {
        'synonym_pattern_count': 0,
        'translation_pattern_count': 0,
        'repeated_words_count': 0,
        'frequent_words_count': 0,
        'sentence_count': 0,
        'avg_sentence_length': 0.0,
        'complex_word_count': 0,
        'complex_word_ratio': 0.0
    }

def extract_bert_sentence_features(text: str) -> dict:
    """
    Extract BERT sentence embedding features (simplified version)
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_bert_sentence_features()
    
    features = {}
    
    # 1. Sentence-level features
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    features['sentence_count'] = len(sentences)
    
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences]
        features['avg_sentence_length'] = np.mean(sentence_lengths)
        features['max_sentence_length'] = np.max(sentence_lengths)
        features['min_sentence_length'] = np.min(sentence_lengths)
        features['sentence_length_std'] = np.std(sentence_lengths)
    else:
        features['avg_sentence_length'] = 0
        features['max_sentence_length'] = 0
        features['min_sentence_length'] = 0
        features['sentence_length_std'] = 0
    
    # 2. Paragraph-level features
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    features['paragraph_count'] = len(paragraphs)
    
    if paragraphs:
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        features['avg_paragraph_length'] = np.mean(paragraph_lengths)
        features['max_paragraph_length'] = np.max(paragraph_lengths)
        features['min_paragraph_length'] = np.min(paragraph_lengths)
        features['paragraph_length_std'] = np.std(paragraph_lengths)
    else:
        features['avg_paragraph_length'] = 0
        features['max_paragraph_length'] = 0
        features['min_paragraph_length'] = 0
        features['paragraph_length_std'] = 0
    
    # 3. Coherence features
    words = text.lower().split()
    if len(words) > 1:
        # Calculate word transition diversity
        bigrams = [words[i] + '_' + words[i+1] for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        features['bigram_diversity'] = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
    else:
        features['bigram_diversity'] = 0
    
    return features

def get_empty_bert_sentence_features() -> dict:
    """Return empty BERT sentence features"""
    return {
        'sentence_count': 0,
        'avg_sentence_length': 0.0,
        'max_sentence_length': 0,
        'min_sentence_length': 0,
        'sentence_length_std': 0.0,
        'paragraph_count': 0,
        'avg_paragraph_length': 0.0,
        'max_paragraph_length': 0,
        'min_paragraph_length': 0,
        'paragraph_length_std': 0.0,
        'bigram_diversity': 0.0
    }

def calculate_advanced_text_processing_features(df: pd.DataFrame, tfidf_models: dict = None) -> pd.DataFrame:
    """
    Calculate all advanced text processing features
    """
    print("Calculating advanced text processing features...")
    
    all_features = []
    
    for idx, row in df.iterrows():
        text = row['comment_text']
        
        # Combine all advanced text processing features
        features = {}
        features.update(extract_advanced_tfidf_features(text, tfidf_models or {}))
        features.update(extract_word_embedding_features(text, {}))
        features.update(extract_text_augmentation_features(text))
        features.update(extract_bert_sentence_features(text))
        
        all_features.append(features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} advanced text processing features")
    return df

# --- Feature Selection & Engineering ---

def extract_feature_interaction_terms(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Extract feature interaction terms (multiplicative and additive combinations)
    """
    print("Calculating feature interaction terms...")
    
    # Select only numerical features for interactions
    numerical_features = []
    for col in feature_columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    interaction_features = {}
    
    # 1. Multiplicative interactions (product of two features)
    for i, feat1 in enumerate(numerical_features[:10]):  # Limit to first 10 to avoid explosion
        for j, feat2 in enumerate(numerical_features[i+1:11]):  # Limit to avoid too many features
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_features[interaction_name] = df[feat1] * df[feat2]
    
    # 2. Additive interactions (sum of two features)
    for i, feat1 in enumerate(numerical_features[:10]):
        for j, feat2 in enumerate(numerical_features[i+1:11]):
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_plus_{feat2}"
                interaction_features[interaction_name] = df[feat1] + df[feat2]
    
    # 3. Ratio interactions (division of two features, with zero handling)
    for i, feat1 in enumerate(numerical_features[:10]):
        for j, feat2 in enumerate(numerical_features[i+1:11]):
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_div_{feat2}"
                # Avoid division by zero
                interaction_features[interaction_name] = df[feat1] / (df[feat2] + 1e-8)
    
    # 4. Difference interactions (absolute difference)
    for i, feat1 in enumerate(numerical_features[:10]):
        for j, feat2 in enumerate(numerical_features[i+1:11]):
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_diff_{feat2}"
                interaction_features[interaction_name] = np.abs(df[feat1] - df[feat2])
    
    # Convert to DataFrame and merge
    if interaction_features:
        interaction_df = pd.DataFrame(interaction_features)
        df = pd.concat([df, interaction_df], axis=1)
        print(f"Added {len(interaction_features)} feature interaction terms")
    else:
        print("No interaction features added")
    
    return df

def calculate_mutual_information_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """
    Calculate mutual information between features and target
    """
    print("Calculating mutual information features...")
    
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping mutual information features.")
        return df
    
    # Select numerical features
    numerical_features = []
    for col in df.columns:
        if col != target_column and col != 'comment_text' and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    if not numerical_features:
        print("No numerical features found for mutual information calculation")
        return df
    
    # Calculate mutual information
    try:
        X = df[numerical_features].fillna(0)
        y = df[target_column].fillna(0)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create features based on mutual information
        mi_features = {}
        
        # 1. Top feature importance scores
        top_features = np.argsort(mi_scores)[-10:]  # Top 10 features
        for i, idx in enumerate(top_features):
            feature_name = numerical_features[idx]
            mi_features[f'mi_score_{feature_name}'] = mi_scores[idx]
        
        # 2. Feature importance ranking
        feature_ranks = np.argsort(np.argsort(mi_scores))
        for i, feature_name in enumerate(numerical_features):
            mi_features[f'mi_rank_{feature_name}'] = feature_ranks[i]
        
        # 3. Normalized mutual information scores
        max_mi = np.max(mi_scores) if np.max(mi_scores) > 0 else 1
        for i, feature_name in enumerate(numerical_features):
            mi_features[f'mi_norm_{feature_name}'] = mi_scores[i] / max_mi
        
        # Convert to DataFrame and merge
        if mi_features:
            mi_df = pd.DataFrame(mi_features)
            df = pd.concat([df, mi_df], axis=1)
            print(f"Added {len(mi_features)} mutual information features")
        else:
            print("No mutual information features added")
            
    except Exception as e:
        print(f"Error calculating mutual information features: {e}")
    
    return df

def calculate_dimensionality_reduction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate dimensionality reduction features using PCA
    """
    print("Calculating dimensionality reduction features...")
    
    # Select numerical features
    numerical_features = []
    for col in df.columns:
        if col != 'comment_text' and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    if len(numerical_features) < 2:
        print("Not enough numerical features for dimensionality reduction")
        return df
    
    try:
        # Prepare data
        X = df[numerical_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        n_components = min(10, len(numerical_features), X.shape[0])  # Limit components
        pca = PCA(n_components=n_components, random_state=42)
        pca_features = pca.fit_transform(X_scaled)
        
        # Create PCA features
        pca_feature_names = [f'pca_component_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_feature_names, index=df.index)
        
        # Add explained variance features
        explained_var_features = {}
        for i in range(n_components):
            explained_var_features[f'pca_explained_var_{i+1}'] = pca.explained_variance_ratio_[i]
        
        # Add cumulative explained variance
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        for i in range(n_components):
            explained_var_features[f'pca_cumulative_var_{i+1}'] = cumulative_var[i]
        
        # Merge all PCA features
        explained_var_df = pd.DataFrame(explained_var_features, index=df.index)
        df = pd.concat([df, pca_df, explained_var_df], axis=1)
        
        print(f"Added {n_components} PCA components and {len(explained_var_features)} variance features")
        
    except Exception as e:
        print(f"Error calculating dimensionality reduction features: {e}")
    
    return df

def calculate_recursive_feature_elimination_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """
    Calculate recursive feature elimination features
    """
    print("Calculating recursive feature elimination features...")
    
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found. Skipping RFE features.")
        return df
    
    # Select numerical features
    numerical_features = []
    for col in df.columns:
        if col != target_column and col != 'comment_text' and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    if len(numerical_features) < 5:
        print("Not enough numerical features for RFE")
        return df
    
    try:
        # Prepare data
        X = df[numerical_features].fillna(0)
        y = df[target_column].fillna(0)
        
        # Limit features for RFE to avoid memory issues
        if len(numerical_features) > 50:
            # Select top 50 features by variance
            feature_vars = X.var()
            top_features = feature_vars.nlargest(50).index.tolist()
            X = X[top_features]
            numerical_features = top_features
        
        # Apply RFE
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        n_features_to_select = min(10, len(numerical_features))
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        
        # Create RFE features
        rfe_features = {}
        
        # 1. Feature selection mask
        for i, feature_name in enumerate(numerical_features):
            rfe_features[f'rfe_selected_{feature_name}'] = 1 if rfe.support_[i] else 0
        
        # 2. Feature ranking
        for i, feature_name in enumerate(numerical_features):
            rfe_features[f'rfe_rank_{feature_name}'] = rfe.ranking_[i]
        
        # 3. Feature importance from RFE
        if hasattr(rfe, 'estimator_') and hasattr(rfe.estimator_, 'feature_importances_'):
            for i, feature_name in enumerate(numerical_features):
                if i < len(rfe.estimator_.feature_importances_):
                    rfe_features[f'rfe_importance_{feature_name}'] = rfe.estimator_.feature_importances_[i]
        
        # Convert to DataFrame and merge
        if rfe_features:
            rfe_df = pd.DataFrame(rfe_features)
            df = pd.concat([df, rfe_df], axis=1)
            print(f"Added {len(rfe_features)} RFE features")
        else:
            print("No RFE features added")
            
    except Exception as e:
        print(f"Error calculating RFE features: {e}")
    
    return df

def calculate_feature_selection_engineering_features(df: pd.DataFrame, target_column: str = 'rule_violation') -> pd.DataFrame:
    """
    Calculate all feature selection and engineering features
    """
    print("Calculating feature selection and engineering features...")
    
    # Get numerical feature columns
    numerical_features = []
    for col in df.columns:
        if col != target_column and col != 'comment_text' and df[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
    
    # 1. Feature interaction terms
    df = extract_feature_interaction_terms(df, numerical_features)
    
    # 2. Mutual information features
    df = calculate_mutual_information_features(df, target_column)
    
    # 3. Dimensionality reduction features
    df = calculate_dimensionality_reduction_features(df)
    
    # 4. Recursive feature elimination features
    df = calculate_recursive_feature_elimination_features(df, target_column)
    
    print("Feature selection and engineering features completed")
    return df

def calculate_rule_specific_comparisons(df: pd.DataFrame, tfidf_model: TfidfVectorizer = None, mean_vectors: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Calculate rule-specific comparison features between positive and negative examples for each rule
    """
    print("Calculating rule-specific comparison features...")
    
    # Check if we have the required columns for rule-specific analysis
    required_cols = ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'rule']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required columns for rule-specific comparisons. Skipping...")
        return df
    
    if tfidf_model is None:
        print("Warning: No TF-IDF model provided for rule-specific comparisons. Skipping...")
        return df
    
    # Get unique rules
    unique_rules = df['rule'].unique()
    print(f"Found {len(unique_rules)} unique rules for comparison analysis")
    
    # Calculate rule-specific patterns for each rule
    rule_patterns = {}
    
    for rule in unique_rules:
        rule_data = df[df['rule'] == rule]
        
        if len(rule_data) < 2:  # Need at least 2 examples to compare
            continue
            
        # Get positive and negative examples for this rule
        pos_examples = []
        neg_examples = []
        
        for _, row in rule_data.iterrows():
            if pd.notna(row['positive_example_1']):
                pos_examples.append(str(row['positive_example_1']))
            if pd.notna(row['positive_example_2']):
                pos_examples.append(str(row['positive_example_2']))
            if pd.notna(row['negative_example_1']):
                neg_examples.append(str(row['negative_example_1']))
            if pd.notna(row['negative_example_2']):
                neg_examples.append(str(row['negative_example_2']))
        
        if len(pos_examples) == 0 or len(neg_examples) == 0:
            continue
            
        # Calculate rule-specific patterns
        rule_patterns[rule] = calculate_rule_patterns(pos_examples, neg_examples, tfidf_model)
    
    # Apply rule-specific features to each row
    all_features = []
    
    for idx, row in df.iterrows():
        rule = row['rule']
        text = row['comment_text']
        
        if rule in rule_patterns:
            features = calculate_rule_specific_features(text, rule_patterns[rule], tfidf_model)
        else:
            features = get_empty_rule_specific_features()
        
        all_features.append(features)
    
    # Convert to DataFrame and merge with original
    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df], axis=1)
    
    print(f"Added {len(features_df.columns)} rule-specific comparison features")
    return df

def calculate_rule_patterns(pos_examples: list, neg_examples: list, tfidf_model: TfidfVectorizer) -> dict:
    """
    Calculate patterns that distinguish positive from negative examples for a specific rule
    """
    # Transform examples to TF-IDF vectors
    pos_vectors = tfidf_model.transform(pos_examples)
    neg_vectors = tfidf_model.transform(neg_examples)
    
    # Calculate median vectors for positive and negative examples
    pos_median = np.median(pos_vectors.toarray(), axis=0)
    neg_median = np.median(neg_vectors.toarray(), axis=0)
    
    # Calculate difference vector (what makes positive examples different from negative)
    difference_vector = pos_median - neg_median
    
    # Calculate variance in positive and negative examples
    pos_variance = np.var(pos_vectors.toarray(), axis=0)
    neg_variance = np.var(neg_vectors.toarray(), axis=0)
    
    # Calculate average similarity within positive and negative examples
    pos_similarities = []
    for i in range(len(pos_examples)):
        for j in range(i+1, len(pos_examples)):
            sim = cosine_similarity(pos_vectors[i:i+1], pos_vectors[j:j+1])[0][0]
            pos_similarities.append(sim)
    
    neg_similarities = []
    for i in range(len(neg_examples)):
        for j in range(i+1, len(neg_examples)):
            sim = cosine_similarity(neg_vectors[i:i+1], neg_vectors[j:j+1])[0][0]
            neg_similarities.append(sim)
    
    return {
        'pos_median': pos_median,
        'neg_median': neg_median,
        'difference_vector': difference_vector,
        'pos_variance': pos_variance,
        'neg_variance': neg_variance,
        'pos_avg_similarity': np.mean(pos_similarities) if pos_similarities else 0,
        'neg_avg_similarity': np.mean(neg_similarities) if neg_similarities else 0,
        'pos_consistency': 1 - np.std(pos_similarities) if pos_similarities else 0,
        'neg_consistency': 1 - np.std(neg_similarities) if neg_similarities else 0
    }

def calculate_rule_specific_features(text: str, rule_patterns: dict, tfidf_model: TfidfVectorizer) -> dict:
    """
    Calculate rule-specific features for a given text based on rule patterns
    """
    if not isinstance(text, str) or not text.strip():
        return get_empty_rule_specific_features()
    
    # Transform text to TF-IDF vector
    text_vector = tfidf_model.transform([text])
    
    features = {}
    
    # 1. Similarity to positive vs negative median vectors
    pos_similarity = cosine_similarity(text_vector, [rule_patterns['pos_median']])[0][0]
    neg_similarity = cosine_similarity(text_vector, [rule_patterns['neg_median']])[0][0]
    
    features['rule_pos_similarity'] = pos_similarity
    features['rule_neg_similarity'] = neg_similarity
    features['rule_similarity_diff'] = pos_similarity - neg_similarity
    features['rule_similarity_ratio'] = pos_similarity / (neg_similarity + 1e-8)
    
    # 2. Alignment with difference vector (what makes positive different from negative)
    diff_alignment = cosine_similarity(text_vector, [rule_patterns['difference_vector']])[0][0]
    features['rule_diff_alignment'] = diff_alignment
    
    # 3. Consistency with positive vs negative patterns
    pos_consistency = 1 - np.std(cosine_similarity(text_vector, [rule_patterns['pos_median']])[0])
    neg_consistency = 1 - np.std(cosine_similarity(text_vector, [rule_patterns['neg_median']])[0])
    
    features['rule_pos_consistency'] = pos_consistency
    features['rule_neg_consistency'] = neg_consistency
    features['rule_consistency_diff'] = pos_consistency - neg_consistency
    
    # 4. Variance alignment (how much the text varies like positive vs negative examples)
    text_variance = np.var(text_vector.toarray(), axis=0)
    pos_var_alignment = cosine_similarity([text_variance], [rule_patterns['pos_variance']])[0][0]
    neg_var_alignment = cosine_similarity([text_variance], [rule_patterns['neg_variance']])[0][0]
    
    features['rule_pos_var_alignment'] = pos_var_alignment
    features['rule_neg_var_alignment'] = neg_var_alignment
    features['rule_var_alignment_diff'] = pos_var_alignment - neg_var_alignment
    
    # 5. Overall rule violation score (combined metric)
    violation_score = (
        0.3 * features['rule_similarity_diff'] +
        0.3 * features['rule_diff_alignment'] +
        0.2 * features['rule_consistency_diff'] +
        0.2 * features['rule_var_alignment_diff']
    )
    features['rule_violation_score'] = violation_score
    
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


# --- Feature Calculation Functions (No Change) ---

def calculate_simple_features(df: pd.DataFrame, scaler: RobustScaler = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """Calculates and scales continuous structural features with enhanced preprocessing."""
    
    # Enhanced text cleaning
    df['comment_text'] = df['comment_text'].astype(str).fillna('')
    df['comment_text'] = df['comment_text'].apply(_clean_and_normalize_text)
    
    # More robust length calculation (character and word count)
    df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))
    df['comment_char_length'] = df['comment_text'].apply(lambda x: len(x))

    # Enhanced exclamation frequency with better normalization
    df['exclamation_frequency'] = df['comment_text'].apply(_get_exclamation_frequency)

    # Additional text quality features
    df['avg_word_length'] = df['comment_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    df['punctuation_ratio'] = df['comment_text'].apply(lambda x: sum(1 for c in x if c in '!?.,;:') / len(x) if len(x) > 0 else 0)
    
    # Features to scale
    continuous_features = ['comment_length', 'comment_char_length', 'exclamation_frequency', 'avg_word_length', 'punctuation_ratio']
    
    # Handle outliers and missing values
    for feature in continuous_features:
        # Replace infinite values with NaN
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median
        df[feature] = df[feature].fillna(df[feature].median())
    
    if scaler is None:
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
    else:
        df[continuous_features] = scaler.transform(df[continuous_features])
    
    return df, scaler

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rule-specific binary interaction features."""
    
    df['legal_advice_interaction_feature'] = df['comment_text'].apply(_check_legal_advice_interaction)
    df['promo_persuasion_feature'] = df['comment_text'].apply(_calculate_promo_persuasion_feature)
    
    return df

def calculate_similarity_features(
    df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates the cosine similarity of each comment to the mean violation and safe vectors.
    """
    
    # Use TRANSFORM to vectorize based on the fitted vocabulary
    X_current_tfidf = tfidf_vectorizer.transform(df['comment_text'])
    
    mean_violation = mean_vectors['violation']
    mean_safe = mean_vectors['safe']
    
    violation_similarity = []
    safe_similarity = []
    
    # Efficiently calculate distances for all vectors
    for vector in X_current_tfidf:
        # Check for zero vectors (happens if comment is filtered entirely by stop words, etc.)
        vector_array = vector.toarray()[0]
        if np.sum(vector_array) == 0:
            # If a comment is empty after vectorization, set similarity to neutral values
            violation_similarity.append(0.0) 
            safe_similarity.append(0.0)
            continue
            
        # NOTE: Using sparse vector support with scipy.spatial.distance.cosine
        dist_to_violation = cosine(vector_array, mean_violation)
        violation_similarity.append(1 - dist_to_violation)

        dist_to_safe = cosine(vector_array, mean_safe)
        safe_similarity.append(1 - dist_to_safe)

    df['similarity_to_violation'] = violation_similarity
    df['similarity_to_safe'] = safe_similarity
    
    # --- NEW FEATURE: Boundary Proximity Score ---
    if 'semantic_difference' in mean_vectors:
        # Retrieve the difference vector
        difference_vec = mean_vectors['semantic_difference']
        
        # Reshape the single difference vector for proper comparison (1 row, N features)
        difference_vec_reshaped = difference_vec.reshape(1, -1)
        
        # Calculate similarity between all comments and the semantic difference vector
        similarity_scores = cosine_similarity(X_current_tfidf, difference_vec_reshaped)
        
        # The final score indicates proximity to the violation concept
        df['boundary_proximity_score'] = similarity_scores.flatten()
        
        print(f"Calculated boundary proximity scores for {len(df)} comments")
    else:
        print("Warning: semantic_difference vector not found in mean_vectors. Skipping boundary proximity score.")
        df['boundary_proximity_score'] = 0.0
    
    return df

def calculate_archetype_vector(texts, tfidf_model):
    """
    Calculate the median vector from a collection of texts using the fitted TF-IDF model.
    This creates an archetype vector that represents the typical characteristics of the text collection.
    """
    if len(texts) == 0:
        # Return zero vector if no texts provided
        return np.zeros(tfidf_model.transform(['']).shape[1])
    
    # Transform texts to TF-IDF vectors
    X_exa = tfidf_model.transform(texts)
    X_exa_dense = X_exa.toarray()
    
    # Calculate median vector
    return np.median(X_exa_dense, axis=0)

def calculate_consistency_features(
    df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates the consistency deviation feature by measuring how much the internal 
    similarity of a specific row's examples deviates from the average consistency 
    found across the entire dataset.
    """
    
    # Check if the required columns exist
    if 'positive_example_1' not in df.columns or 'positive_example_2' not in df.columns:
        print("Warning: positive_example_1 or positive_example_2 columns not found. Skipping consistency features.")
        return df
    
    # 1. Transform BOTH text columns using the FITTED tfidf_vectorizer
    text_1 = df['positive_example_1'].fillna('')
    text_2 = df['positive_example_2'].fillna('')

    X_ex1 = tfidf_vectorizer.transform(text_1)
    X_ex2 = tfidf_vectorizer.transform(text_2)

    # 2. Normalize and compute row-wise Cosine Similarity
    X_ex1_norm = normalize(X_ex1, norm='l2', axis=1)
    X_ex2_norm = normalize(X_ex2, norm='l2', axis=1)

    # Calculate cosine similarity between the two positive examples
    consistency_scores = (X_ex1_norm.multiply(X_ex2_norm)).sum(axis=1)
    df['example_consistency'] = np.asarray(consistency_scores).flatten()
    
    # 3. Calculate Global Statistics and Apply Scaling
    if 'consistency_mean' not in mean_vectors:
        # Training phase: Calculate and save the mean and standard deviation
        consistency_mean = df['example_consistency'].mean()
        consistency_std = df['example_consistency'].std()
        
        # Store these statistics in the mean_vectors dictionary
        mean_vectors['consistency_mean'] = consistency_mean
        mean_vectors['consistency_std'] = consistency_std
        
        print(f"Training: Calculated consistency_mean={consistency_mean:.4f}, consistency_std={consistency_std:.4f}")
    else:
        # Validation/Test phase: Load the saved mean and std dev
        consistency_mean = mean_vectors['consistency_mean']
        consistency_std = mean_vectors['consistency_std']
        
        print(f"Validation/Test: Using consistency_mean={consistency_mean:.4f}, consistency_std={consistency_std:.4f}")
    
    # Calculate the final feature (Z-score)
    if consistency_std > 0:  # Avoid division by zero
        df['consistency_deviation'] = (df['example_consistency'] - consistency_mean) / consistency_std
    else:
        df['consistency_deviation'] = 0.0  # If std is 0, set all values to 0
    
    return df

# --- Master Preprocessing Function (The fix is here) ---

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
        
        # B. Calculate Mean Vectors
        violation_mask = df['rule_violation'] == 1 
        
        MEAN_VIOLATION_VECTOR = X_tfidf[violation_mask].mean(axis=0)
        MEAN_SAFE_VECTOR = X_tfidf[~violation_mask].mean(axis=0)
        
        # C. Calculate Semantic Difference Vector (Boundary Proximity Feature)
        # Extract positive and negative example texts
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

    # CRITICAL FIX: Define df_final before returning it
    df_final = df[columns_to_keep] 

    print(f"Preprocessing complete. Final features: {list(df_final.columns)}")
    
    return df_final, tfidf_model, mean_vectors, scaler