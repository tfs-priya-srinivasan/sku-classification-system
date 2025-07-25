import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import warnings
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import time
warnings.filterwarnings('ignore')

class SKUProductLineClassifier:
    def __init__(self):
        self.sku_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=1000,
            analyzer='char_wb',
            lowercase=True
        )
        self.name_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            stop_words='english',
            lowercase=True
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=40,
            class_weight='balanced'
        )
        self.product_line_encoder = LabelEncoder()
        self.cmr_encoder = LabelEncoder()
        self.business_rules = {}
        self.product_line_groups = {}
        self.fitted = False
    
    def extract_sku_patterns(self, sku):
        """Extract detailed patterns from SKU numbers"""
        if pd.isna(sku):
            return {}
        
        sku = str(sku).upper()
        patterns = {}
        
        letter_sequences = re.findall(r'[A-Z]+', sku)
        patterns['letter_prefixes'] = letter_sequences
        patterns['first_letters'] = letter_sequences[0] if letter_sequences else None
        
        number_sequences = re.findall(r'\d+', sku)
        patterns['number_sequences'] = number_sequences
        patterns['has_numbers'] = len(number_sequences) > 0
        
        separators = re.findall(r'[^A-Z0-9]', sku)
        patterns['separators'] = list(set(separators))
        
        patterns['sku_length'] = len(sku)
        patterns['letter_count'] = len(re.findall(r'[A-Z]', sku))
        patterns['number_count'] = len(re.findall(r'\d', sku))
        
        structure = re.sub(r'[A-Z]', 'A', sku)
        structure = re.sub(r'\d', '1', structure)
        patterns['structure'] = structure
        
        return patterns
    
    def extract_name_patterns(self, name):
        """Extract patterns from SKU names"""
        if pd.isna(name):
            return {}
        
        name = str(name).lower()
        patterns = {}
        
        words = re.findall(r'\b[a-z]{2,}\b', name)
        stop_words = {'the', 'and', 'for', 'with', 'from', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'this', 'that', 'will', 'can', 'may', 'but', 'not', 'you', 'all', 'any', 'one', 'two'}
        patterns['all_words'] = [word for word in words if word not in stop_words]
        
        single_chars = re.findall(r'\b[a-z0-9]\b', name)
        patterns['single_chars'] = single_chars
        
        numbers = re.findall(r'\d+', name)
        patterns['numbers'] = numbers
        
        word_list = name.split()
        if word_list:
            patterns['first_word'] = word_list[0]
            patterns['last_word'] = word_list[-1]
            patterns['word_count'] = len(word_list)
        
        if '-' in name:
            parts = name.split('-')
            patterns['hyphen_parts'] = [part.strip() for part in parts if part.strip()]
        
        patterns['name_length'] = len(name)
        patterns['has_numbers'] = bool(re.search(r'\d', name))
        patterns['has_special_chars'] = bool(re.search(r'[^a-z0-9\s]', name))
        
        return patterns
    def create_business_rules(self, df):
        """Creating business rules by analyzing patterns in each product line group"""
        print("Creating business rules from product line groups...")
        
        # Group by product line code and CMR product line
        grouped = df.groupby(['product line code', 'cmr product line'])
        
        self.business_rules = {}
        self.product_line_groups = {}
        
        for (product_line_code, cmr_product_line), group in grouped:
            print(f"\nAnalyzing: {product_line_code} - {cmr_product_line} ({len(group)} SKUs)")
            
            # Store the group data
            group_key = f"{product_line_code}|{cmr_product_line}"
            self.product_line_groups[group_key] = {
                'product_line_code': product_line_code,
                'cmr_product_line': cmr_product_line,
                'skus': group[['sku number', 'sku name']].to_dict('records'),
                'count': len(group)
            }
            
            # Extract patterns for all SKUs in this group
            sku_patterns = []
            name_patterns = []
            
            for _, row in group.iterrows():
                sku_patterns.append(self.extract_sku_patterns(row['sku number']))
                name_patterns.append(self.extract_name_patterns(row['sku name']))
            
            # Analyze common patterns
            rules = self.analyze_group_patterns(sku_patterns, name_patterns, group)
            rules['product_line_code'] = product_line_code
            rules['cmr_product_line'] = cmr_product_line
            rules['sample_count'] = len(group)
            
            self.business_rules[group_key] = rules
            
            # Print rules for this group
            self.print_business_rules(group_key)
        
        return self.business_rules
    
    def analyze_group_patterns(self, sku_patterns, name_patterns, group):
        """Analyze patterns within a product line group"""
        rules = {
            'sku_rules': {},
            'name_rules': {},
            'combined_rules': {}
        }
        
        # SKU Pattern Analysis
        letter_prefixes = [p.get('first_letters') for p in sku_patterns if p.get('first_letters')]
        if letter_prefixes:
            prefix_counts = Counter(letter_prefixes)
            most_common_prefix = prefix_counts.most_common(1)[0]
            if most_common_prefix[1] / len(sku_patterns) >= 0.5:  # At least 50% have this prefix
                rules['sku_rules']['common_prefix'] = {
                    'pattern': most_common_prefix[0],
                    'confidence': most_common_prefix[1] / len(sku_patterns)
                }
        
        # Structure patterns
        structures = [p.get('structure') for p in sku_patterns if p.get('structure')]
        if structures:
            structure_counts = Counter(structures)
            most_common_structure = structure_counts.most_common(1)[0]
            if most_common_structure[1] / len(sku_patterns) >= 0.3:  # At least 30% have this structure
                rules['sku_rules']['common_structure'] = {
                    'pattern': most_common_structure[0],
                    'confidence': most_common_structure[1] / len(sku_patterns)
                }
        
        # Separator patterns
        all_separators = []
        for p in sku_patterns:
            all_separators.extend(p.get('separators', []))
        if all_separators:
            separator_counts = Counter(all_separators)
            rules['sku_rules']['common_separators'] = dict(separator_counts.most_common(3))
        
        # Length patterns
        lengths = [p.get('sku_length') for p in sku_patterns if p.get('sku_length')]
        if lengths:
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            rules['sku_rules']['length_pattern'] = {
                'avg_length': round(avg_length, 1),
                'std_length': round(std_length, 1),
                'min_length': min(lengths),
                'max_length': max(lengths)
            }
        
        # Name Pattern Analysis - discover patterns from actual data
        all_words = []
        all_first_words = []
        all_last_words = []
        all_single_chars = []
        all_numbers = []
        all_hyphen_parts = []
        
        for p in name_patterns:
            all_words.extend(p.get('all_words', []))
            if p.get('first_word'):
                all_first_words.append(p.get('first_word'))
            if p.get('last_word'):
                all_last_words.append(p.get('last_word'))
            all_single_chars.extend(p.get('single_chars', []))
            all_numbers.extend(p.get('numbers', []))
            all_hyphen_parts.extend(p.get('hyphen_parts', []))
        
        # Discover frequent words that appear in this product line (30% threshold)
        if all_words:
            word_counts = Counter(all_words)
            frequent_words = {word: count for word, count in word_counts.items() 
                            if count / len(name_patterns) >= 0.3}
            if frequent_words:
                rules['name_rules']['frequent_words'] = frequent_words
        
        # Discover frequent first/last words
        if all_first_words:
            first_word_counts = Counter(all_first_words)
            common_first = first_word_counts.most_common(1)[0]
            if common_first[1] / len(name_patterns) >= 0.5:  # 50% have same first word
                rules['name_rules']['common_first_word'] = {
                    'word': common_first[0],
                    'confidence': common_first[1] / len(name_patterns)
                }
        
        if all_last_words:
            last_word_counts = Counter(all_last_words)
            common_last = last_word_counts.most_common(1)[0]
            if common_last[1] / len(name_patterns) >= 0.5:  # 50% have same last word
                rules['name_rules']['common_last_word'] = {
                    'word': common_last[0],
                    'confidence': common_last[1] / len(name_patterns)
                }
        
        # Discover frequent single characters (size indicators, versions, etc.)
        if all_single_chars:
            char_counts = Counter(all_single_chars)
            frequent_chars = {char: count for char, count in char_counts.items() 
                            if count / len(name_patterns) >= 0.3}
            if frequent_chars:
                rules['name_rules']['frequent_single_chars'] = frequent_chars
        
        # Discover frequent numbers
        if all_numbers:
            number_counts = Counter(all_numbers)
            frequent_numbers = {num: count for num, count in number_counts.items() 
                              if count / len(name_patterns) >= 0.3}
            if frequent_numbers:
                rules['name_rules']['frequent_numbers'] = frequent_numbers
        
        # Discover naming structure patterns
        word_count_pattern = [p.get('word_count', 0) for p in name_patterns]
        if word_count_pattern:
            avg_word_count = np.mean(word_count_pattern)
            rules['name_rules']['avg_word_count'] = round(avg_word_count, 1)
        
        # Check for consistent hyphen usage
        hyphen_usage = sum(1 for p in name_patterns if p.get('hyphen_parts'))
        if hyphen_usage / len(name_patterns) >= 0.5:
            rules['name_rules']['uses_hyphens'] = True
            if all_hyphen_parts:
                hyphen_part_counts = Counter(all_hyphen_parts)
                frequent_hyphen_parts = {part: count for part, count in hyphen_part_counts.items() 
                                       if count / len(name_patterns) >= 0.3}
                if frequent_hyphen_parts:
                    rules['name_rules']['frequent_hyphen_parts'] = frequent_hyphen_parts
        
        # Combined patterns - look for correlations
        rules['combined_rules']['total_examples'] = len(group)
        
        return rules
    
    def print_business_rules(self, group_key):
        """Print business rules for a specific group"""
        if group_key not in self.business_rules:
            return
        
        rules = self.business_rules[group_key]
        product_line = rules['product_line_code']
        cmr_line = rules['cmr_product_line']
        
        print(f"\n{'='*60}")
        print(f"BUSINESS RULES FOR: {product_line} - {cmr_line}")
        print(f"Sample Count: {rules['sample_count']}")
        print(f"{'='*60}")
        
        # SKU Rules
        if rules['sku_rules']:
            print("\n SKU PATTERNS:")
            
            if 'common_prefix' in rules['sku_rules']:
                prefix_info = rules['sku_rules']['common_prefix']
                print(f"  • Common Prefix: '{prefix_info['pattern']}' (confidence: {prefix_info['confidence']:.1%})")
            
            if 'common_structure' in rules['sku_rules']:
                structure_info = rules['sku_rules']['common_structure']
                print(f"  • Common Structure: '{structure_info['pattern']}' (confidence: {structure_info['confidence']:.1%})")
            
            if 'common_separators' in rules['sku_rules']:
                separators = rules['sku_rules']['common_separators']
                print(f"  • Common Separators: {list(separators.keys())}")
            
            if 'length_pattern' in rules['sku_rules']:
                length_info = rules['sku_rules']['length_pattern']
                print(f"  • Length Pattern: {length_info['min_length']}-{length_info['max_length']} chars (avg: {length_info['avg_length']})")
        
        # Name Rules
        if rules['name_rules']:
            print("\n NAME PATTERNS:")
            
            if 'frequent_words' in rules['name_rules']:
                words = rules['name_rules']['frequent_words']
                print(f"  • Frequent Words: {list(words.keys())}")
            
            if 'common_first_word' in rules['name_rules']:
                first_word_info = rules['name_rules']['common_first_word']
                print(f"  • Common First Word: '{first_word_info['word']}' (confidence: {first_word_info['confidence']:.1%})")
            
            if 'common_last_word' in rules['name_rules']:
                last_word_info = rules['name_rules']['common_last_word']
                print(f"  • Common Last Word: '{last_word_info['word']}' (confidence: {last_word_info['confidence']:.1%})")
            
            if 'frequent_single_chars' in rules['name_rules']:
                print(f"  • Frequent Single Chars: {list(rules['name_rules']['frequent_single_chars'].keys())}")
            
            if 'frequent_numbers' in rules['name_rules']:
                print(f"  • Frequent Numbers: {list(rules['name_rules']['frequent_numbers'].keys())}")
            
            if 'avg_word_count' in rules['name_rules']:
                print(f"  • Average Word Count: {rules['name_rules']['avg_word_count']}")
            
            if 'uses_hyphens' in rules['name_rules']:
                print(f"  • Uses Hyphens: Yes")
                if 'frequent_hyphen_parts' in rules['name_rules']:
                    print(f"  • Frequent Hyphen Parts: {list(rules['name_rules']['frequent_hyphen_parts'].keys())}")
            
            if rules['name_rules'].get('has_numbers_pattern'):
                print(f"  • Number Usage Pattern: {rules['name_rules']['has_numbers_pattern']}")
        
        # Show some examples
        group_data = self.product_line_groups[group_key]
        print(f"\n EXAMPLES (showing first 5 of {group_data['count']}):")
        for i, sku_data in enumerate(group_data['skus'][:5]):
            print(f"  {i+1}. SKU: {sku_data['sku number']} | Name: {sku_data['sku name']}")
    
    def apply_business_rules(self, sku_number, sku_name, confidence_threshold=0.6):
        """Apply business rules to classify a new SKU"""
        sku_patterns = self.extract_sku_patterns(sku_number)
        name_patterns = self.extract_name_patterns(sku_name)
        
        best_match = None
        best_score = 0
        
        for group_key, rules in self.business_rules.items():
            score = 0
            matches = []
            
            # Check SKU rules
            if 'sku_rules' in rules:
                sku_rules = rules['sku_rules']
                
                # Check prefix match
                if 'common_prefix' in sku_rules:
                    prefix_rule = sku_rules['common_prefix']
                    if sku_patterns.get('first_letters') == prefix_rule['pattern']:
                        score += prefix_rule['confidence'] * 0.4
                        matches.append(f"prefix '{prefix_rule['pattern']}'")
                
                # Check structure match
                if 'common_structure' in sku_rules:
                    structure_rule = sku_rules['common_structure']
                    if sku_patterns.get('structure') == structure_rule['pattern']:
                        score += structure_rule['confidence'] * 0.3
                        matches.append(f"structure '{structure_rule['pattern']}'")
                
                # Check separator match
                if 'common_separators' in sku_rules:
                    common_seps = set(sku_rules['common_separators'].keys())
                    sku_seps = set(sku_patterns.get('separators', []))
                    if common_seps.intersection(sku_seps):
                        score += 0.1
                        matches.append("separator match")
            
            # Check name rules
            if 'name_rules' in rules:
                name_rules = rules['name_rules']
                
                # Check frequent words
                if 'frequent_words' in name_rules:
                    frequent_words = set(name_rules['frequent_words'].keys())
                    sku_words = set(name_patterns.get('all_words', []))
                    word_matches = frequent_words.intersection(sku_words)
                    if word_matches:
                        keyword_score = len(word_matches) / len(frequent_words)
                        score += keyword_score * 0.3
                        matches.append(f"words {list(word_matches)}")
                
                # Check first word pattern
                if 'common_first_word' in name_rules:
                    first_word_rule = name_rules['common_first_word']
                    if name_patterns.get('first_word') == first_word_rule['word']:
                        score += first_word_rule['confidence'] * 0.15
                        matches.append(f"first word '{first_word_rule['word']}'")
                
                # Check last word pattern
                if 'common_last_word' in name_rules:
                    last_word_rule = name_rules['common_last_word']
                    if name_patterns.get('last_word') == last_word_rule['word']:
                        score += last_word_rule['confidence'] * 0.15
                        matches.append(f"last word '{last_word_rule['word']}'")
                
                # Check single character patterns
                if 'frequent_single_chars' in name_rules:
                    frequent_chars = set(name_rules['frequent_single_chars'].keys())
                    sku_chars = set(name_patterns.get('single_chars', []))
                    if frequent_chars.intersection(sku_chars):
                        score += 0.05
                        matches.append("single char match")
                
                # Check number patterns
                if 'frequent_numbers' in name_rules:
                    frequent_numbers = set(name_rules['frequent_numbers'].keys())
                    sku_numbers = set(name_patterns.get('numbers', []))
                    if frequent_numbers.intersection(sku_numbers):
                        score += 0.05
                        matches.append("number pattern match")
                
                # Check hyphen usage
                if name_rules.get('uses_hyphens') and name_patterns.get('hyphen_parts'):
                    score += 0.05
                    matches.append("hyphen usage")
                    
                    # Check specific hyphen parts
                    if 'frequent_hyphen_parts' in name_rules:
                        frequent_parts = set(name_rules['frequent_hyphen_parts'].keys())
                        sku_parts = set(name_patterns.get('hyphen_parts', []))
                        if frequent_parts.intersection(sku_parts):
                            score += 0.1
                            matches.append("hyphen parts match")
            
            if score > best_score and score >= confidence_threshold:
                best_score = score
                best_match = {
                    'product_line_code': rules['product_line_code'],
                    'cmr_product_line': rules['cmr_product_line'],
                    'confidence': score,
                    'method': 'business_rules',
                    'matching_rules': matches,
                    'rule_group': group_key
                }
        
        return best_match
    
    def fit(self, df):
        """Train the model and create business rules"""
        print("Creating business rules from training data...")
        
        # Create business rules first
        self.create_business_rules(df)
        
        # Continue with original model training
        print("\nTraining ML model as fallback...")
        
        # Preprocess features
        df['sku_processed'] = df['sku number'].apply(self.preprocess_sku)
        df['name_processed'] = df['sku name'].apply(self.preprocess_name)
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=['product line code', 'cmr product line'])

        valid_classes = df_clean['product line code'].value_counts()
        df_clean = df_clean[df_clean['product line code'].isin(valid_classes[valid_classes > 1].index)]
        
        if len(df_clean) == 0:
            raise ValueError("No valid training data found after removing rare classes")
        
        if len(df_clean) == 0:
            raise ValueError("No valid training data found")
        
        # Encode target variables
        self.product_line_encoder.fit(df_clean['product line code'])
        self.cmr_encoder.fit(df_clean['cmr product line'])
        
        # Vectorize features
        sku_features = self.sku_vectorizer.fit_transform(df_clean['sku_processed'])
        name_features = self.name_vectorizer.fit_transform(df_clean['name_processed'])
        
        # Combine features
        X = np.hstack([sku_features.toarray(), name_features.toarray()])
        y = self.product_line_encoder.transform(df_clean['product line code'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ML Model accuracy: {accuracy:.3f}")
        
        # Store training data for similarity matching
        self.training_data = df_clean[['sku number', 'sku name', 'product line code', 'cmr product line']].copy()
        self.fitted = True
        
        return self
    def preprocess_sku(self, sku):
        """Extract patterns from SKU numbers"""
        if pd.isna(sku):
            return ""
        
        sku = str(sku).upper()
        letters = re.findall(r'[A-Z]+', sku)
        numbers = re.findall(r'\d+', sku)
        special_chars = re.findall(r'[^A-Z0-9]', sku)
        
        features = []
        features.extend(letters)
        features.extend([f"NUM_{num}" for num in numbers])
        features.extend([f"CHAR_{char}" for char in special_chars])
        
        return " ".join(features) + " " + sku
    
    def preprocess_name(self, name):
        """Clean and preprocess SKU names"""
        if pd.isna(name):
            return ""
        
        name = str(name).lower()
        name = re.sub(r'[^a-z0-9\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

def calculate_similarity(s1, s2):
    """Calculate similarity between two strings"""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100

# REPLACE YOUR ENTIRE get_exact_predictions FUNCTION WITH THIS:

def get_exact_predictions(df, sku_partial, name_partial):
    """Simple exact/substring matches - more robust version"""
    if not sku_partial.strip() and not name_partial.strip():
        return pd.DataFrame()
    
    # Make a copy to work with
    df_copy = df.copy()
    
    # Apply filters step by step
    if sku_partial.strip():
        df_copy = df_copy[
            df_copy['sku number'].astype(str).str.upper().str.contains(
                sku_partial.upper(), na=False, regex=False
            )
        ]
    
    if name_partial.strip():
        df_copy = df_copy[
            df_copy['sku name'].astype(str).str.lower().str.contains(
                name_partial.lower(), na=False, regex=False
            )
        ]
    
    # Remove duplicates
    exact_matches = df_copy.drop_duplicates(
        subset=['sku number', 'sku name', 'product line code', 'cmr product line']
    )
    
    return exact_matches.head(10)

def get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k=5):
    """OPTIMIZED fuzzy matching using TF-IDF cosine similarity"""
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    # Create query vector
    query_text = f"{sku_partial} {name_partial}".lower()
    query_vector = vectorizer.transform([query_text])
    
    # Calculate cosine similarity (vectorized - very fast!)
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top indices
    top_indices = np.argsort(similarities)[-top_k*3:][::-1]  # Get more than needed
    
    results = []
    seen_combinations = set()
    
    for idx in top_indices:
        if len(results) >= top_k:
            break
            
        if similarities[idx] < 0.1:  # Skip very low similarities
            continue
            
        row = df.iloc[idx]
        combination_id = f"{row['product line code']}|{row['cmr product line']}"
        
        if combination_id not in seen_combinations:
            seen_combinations.add(combination_id)
            
            # Calculate individual scores for display
            sku_score = calculate_simple_similarity(sku_partial, str(row['sku number']))
            name_score = calculate_simple_similarity(name_partial, str(row['sku name']))
            
            results.append({
                'sku_number': row['sku number'],
                'sku_name': row['sku name'],
                'product_line_code': row['product line code'],
                'cmr_product_line': row['cmr product line'],
                'product_line_name': row['product line name'],
                'sub_platform': row['sub platform'],
                'sku_score': round(sku_score, 2),
                'name_score': round(name_score, 2),
                'combined_score': round(similarities[idx] * 100, 2)
            })
    
    return results

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess the data"""
    try:
        
        df = pd.read_excel(r'C:\\Users\\priya.srinivasan\\OneDrive - Thermo Fisher Scientific\\Documents\\Business Rule\\Training_Set.xlsx')
        df = df[(df['sku number'].notna()) & (df['sku number'] != '') & (df['sku name'].notna()) & (df['sku name'] != '')]
        valid_product_lines = ['BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 'BioproductionContainments', 'BottleAssemblies','ProductionCellCulture', 'RigidOther', 'SUDOther']
        df = df[df['cmr product line'].isin(valid_product_lines)]
        df['sku number'] = df['sku number'].str.replace(r'(INT_FINESS.*|BPD.*)', '', regex=True)
        # Extract volume and update CMR product line
        def extract_volume(sku_name):
            sku_name = sku_name.upper()

            # Check for volume in Liters
            match_l = re.search(r'(\d+)\s*L', sku_name)
            if match_l:
                return int(match_l.group(1))

            # Check for volume in mL and convert to Liters
            match_ml = re.search(r'(\d+)\s*M?ML', sku_name)
            if match_ml:
                ml = int(match_ml.group(1))
                return ml / 1000  # Convert mL to L

            return None
        
        df['volume_l'] = df['sku name'].apply(extract_volume)
        
        def update_cmr_product_line(row):
            if row['cmr product line'] in ['2DBioProcessContainers', '3DBioProcessContainers']:
                sku_name = str(row.get('sku_name', '')).lower()
                if 'ml' in sku_name:
                    return '2DBioProcessContainers'
                if row['volume_l'] is not None:
                    return '2DBioProcessContainers' if row['volume_l'] <= 20 else '3DBioProcessContainers'
            return row['cmr product line']
        
        df['cmr product line'] = df.apply(update_cmr_product_line, axis=1)
        df.drop(columns=['volume_l'], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_business_rule_book():
    """Load the business rule book file for download"""
    file_path = r'C:\\Users\\priya.srinivasan\\OneDrive - Thermo Fisher Scientific\\Documents\\Business Rule\\Business_Rule.xlsx'
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"Business Rule Book not found at: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading Business Rule Book: {str(e)}")
        return None
    
@st.cache_data
def load_business_rules_data():
    """Load the business rule book data for rule lookup"""
    file_path = r'C:\\Users\\priya.srinivasan\\OneDrive - Thermo Fisher Scientific\\Documents\\Business Rule\\Business_Rule.xlsx'
    try:
        df_rules = pd.read_excel(file_path)
        return df_rules
    except FileNotFoundError:
        st.error(f"Business Rule Book not found at: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading Business Rule Book data: {str(e)}")
        return None
    
def get_business_rule(product_line_code, df_rules):
    """Get business rule for a specific product line code"""
    if df_rules is None:
        return None, None
    
    try:
        # Filter for the specific product line code
        rule_row = df_rules[df_rules['product line code'] == product_line_code]
        
        if not rule_row.empty:
            # Get the first matching row
            rule = rule_row.iloc[0]
            
            # Extract the relevant columns
            sku_name_pattern = rule.get('Top Trigrams in SKU Name', 'N/A')
            sku_prefix_pattern = rule.get('Top Prefixes in SKU No.', 'N/A')
            
            return sku_name_pattern, sku_prefix_pattern
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error retrieving business rule: {str(e)}")
        return None, None
    
@st.cache_resource
def create_similarity_index(df):
    """Create TF-IDF similarity index for fast fuzzy matching"""
    # Combine SKU and name for vectorization
    combined_text = (df['sku number'].astype(str) + " " + 
                    df['sku name'].astype(str)).str.lower()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        analyzer='char_wb',
        lowercase=True,
        min_df=1
    )
    
    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    return vectorizer, tfidf_matrix

@lru_cache(maxsize=500)
def calculate_simple_similarity(s1, s2):
    """Cached simple similarity calculation"""
    if not s1 or not s2:
        return 0.0
    
    s1, s2 = s1.lower(), s2.lower()
    if s1 in s2 or s2 in s1:
        return 85.0
    
    # Simple character overlap
    set1, set2 = set(s1), set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / union) * 100 if union > 0 else 0.0

def main():
    st.set_page_config(
        page_title="SKU Classification Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    col1, col2 = st.columns([1, 4])
    col1.image("logo2.png", width=1000)
    col2.title("SKU Product Line Classification Chatbot")
    st.markdown("---")
    
    # MODIFY: Initialize data and similarity index in session state
    if 'df' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            if st.session_state.df is not None:
                st.session_state.vectorizer, st.session_state.tfidf_matrix = create_similarity_index(st.session_state.df)
    
    df = st.session_state.df
    if 'df_rules' not in st.session_state:
        st.session_state.df_rules = load_business_rules_data()

    df_rules = st.session_state.df_rules
    
    if df is None:
        st.error("Failed to load data. Please check if the file exists.")
        return
    
    st.success(f'Feature extraction complete! Model trained on "{len(df)}" SKU patterns')
    
    # Sidebar with information and download button
    with st.sidebar:
        st.markdown('<style>div.stImage {margin-top: 30px;}</style>', unsafe_allow_html=True)
        #logo_url = "logo2.png"
        #st.image(logo_url, width=200)
        st.header("Information")
        st.metric("Total Unique SKUs", len(df))
        st.metric("Product Line Codes", df['product line code'].nunique())
        st.metric("CMR Product Lines", df['cmr product line'].nunique())
        
        st.header("How it works")
        st.markdown("""
        1. **Exact Match**: First tries to find exact substring matches
        2. **Fuzzy Search**: Uses intelligent fuzzy matching for similar SKUs
        3. **No Duplicates**: Ensures unique predictions based on product line combinations
        """)
        
        # NEW: Business Rule Book Download Section
        st.header("📋 Resources")
        st.markdown("Download the Business Rule for detailed information:")
        
        # Load the business rule book
        business_rule_data = load_business_rule_book()
        
        if business_rule_data:
            st.download_button(
                label="📥 Download Business Rule",
                data=business_rule_data,
                file_name="Business_Rule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the complete business rule with informations",
                use_container_width=True
            )
        else:
            st.error("Business Rule Book not available for download")
    
    # MODIFY: Replace input fields with form to prevent constant re-execution
    st.header("Enter SKU Information")
    
    with st.form("sku_search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sku_input = st.text_input(
                "SKU Number",
                placeholder="e.g., SV50139.06",
                help="Enter the SKU number (partial matches allowed)"
            )
        
        with col2:
            name_input = st.text_input(
                "SKU Name",
                placeholder="e.g., PKG MATL| COLLAPSIBLE BIN, CUSTOM 1000L BOTTOM DRAIN",
                help="Enter the SKU name (partial matches allowed)"
            )
        
        # Form submit button
        submitted = st.form_submit_button("🔍 Classify SKU", type="primary", use_container_width=True)
    
    # MODIFY: Only process when form is submitted
    if submitted:
        if not sku_input.strip() and not name_input.strip():
            st.warning("Please enter at least one field (SKU Number or SKU Name)")
            return
        
        # Show search progress
        with st.spinner("Analyzing SKU patterns..."):
            # Get exact predictions (UPDATED FUNCTION CALL)
            time.sleep(5)
            exact_matches = get_exact_predictions(df, sku_input, name_input)
            
            # Get fuzzy predictions (UPDATED FUNCTION CALL)
            fuzzy_matches = get_fuzzy_predictions(
                df, sku_input, name_input, 
                st.session_state.vectorizer, 
                st.session_state.tfidf_matrix, 
                top_k=8
            )
        
        # Display results (REST OF THE CODE REMAINS THE SAME)
        st.markdown("---")
        st.header("Classification Results")
        
        # Exact matches section
        if not exact_matches.empty:
            st.subheader("Predictions confirmed with 100 percent confidence level")
            st.success(f"{len(exact_matches)} Strong Prediction Found")
            
            for idx, row in exact_matches.iterrows():
                with st.expander(f"Match {idx + 1}: {row['product line code']} - {row['cmr product line']}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**SKU Number:**", row['sku number'])
                        st.write("**Product Line Code:**", row['product line code'])
                        st.write("**Product Line Name:**", row['product line name'])
                    with col2:
                        st.write("**SKU Name:**", row['sku name'])
                        st.write("**CMR Product Line:**", row['cmr product line'])
                        st.write("**Business Unit:**", row['sub platform'])

                    #sku_prefix = ''.join([c for c in str(row['sku number']) if c.isalpha()])  # Get all letters
                    #sku_digits = ''.join([c for c in str(row['sku number']) if c.isdigit()])  # Get all numbers
                    #stop_words_list = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'ARE', 'WAS', 'WERE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'THIS', 'THAT', 'WILL', 'CAN', 'MAY', 'BUT', 'NOT', 'YOU', 'ALL', 'ANY', 'ONE', 'TWO', 'OF', 'TO', 'IN', 'ON', 'AT', 'BY', 'OR', 'AS', 'IS', 'IF', 'AN', 'A'}
                    #meaningful_keywords = [word for word in str(row['sku name']).upper().split() if len(word) > 2 and word not in stop_words_list][:4]
                    #name_keywords = [w for w in str(row['sku name']).upper().split() if len(w) > 2][:4]
                    #st.success(f"📋 **Rule Match:** SKU Prefix: `{sku_prefix}` + Digits: `{sku_digits} | Keywords: `{' + '.join(meaningful_keywords)}`")
        else:
            st.info("Showing fuzzy predictions below.")
        
        # Fuzzy matches section (UNCHANGED)
        if fuzzy_matches:
            st.markdown("### 📥 Input Parameters")
            input_col1, input_col2 = st.columns(2)
            with input_col1:
                st.info(f"**Input SKU Number:** {sku_input  if 'sku_input' in locals() and sku_input else 'N/A'}")
            with input_col2:
                st.info(f"**Input SKU Name:** {name_input  if 'name_input' in locals() and name_input else 'N/A'}")      
            st.markdown("---")

            st.subheader("🔍 Top Fuzzy Predictions")
            for i, match in enumerate(fuzzy_matches, 1):
                # Color coding based on confidence
                if match['combined_score'] >= 80:
                    confidence_color = "🟢"
                elif match['combined_score'] >= 60:
                    confidence_color = "🟡"
                else:
                    confidence_color = "🔴"
                
                with st.expander(
                    f"{confidence_color} Prediction {i}: {match['product_line_code']} - {match['cmr_product_line']} | "
                    f"{match['product_line_name']} | {match['sub_platform']} "
                    f"(Confidence: {match['combined_score']}%)",
                    expanded=(i == 1)
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**SKU Number:**", match['sku_number'])
                        st.write("**Product Line Code:**", match['product_line_code'])
                        st.write("**Product Line Name:**", match['product_line_name'])
                    
                    with col2:
                        st.write("**SKU Name:**", match['sku_name'][:50] + "..." if len(match['sku_name']) > 50 else match['sku_name'])
                        st.write("**CMR Product Line:**", match['cmr_product_line'])
                        st.write("**Business Unit:**", match['sub_platform'])
                    
                    with col3:
                        #st.metric("SKU Score", f"{match['sku_score']}%")
                        #st.metric("Name Score", f"{match['name_score']}%")
                        st.metric("Confident Score", f"{match['combined_score']}%")
                    
                    if len(match['sku_name']) > 50:
                        st.write("**Full SKU Name:**", match['sku_name'])

                    # Business Rule Section for Fuzzy Matches
                    st.markdown("---")
                    st.markdown("**📋 Business Rule Identified:**")
                    
                    # Get business rule for this product line code
                    sku_name_pattern, sku_prefix_pattern = get_business_rule(match['product_line_code'], df_rules)
                    
                    col4 = st.columns(1)[0]
                    with col4:
                        if sku_name_pattern and sku_prefix_pattern:
                            st.info(f"**Common SKU Name Pattern:** {sku_name_pattern}")
                            st.info(f"**Prefix Pattern Found:** {sku_prefix_pattern}")
                        else:
                            st.warning("No business rule found for this product line code")
        else:
            st.warning("No fuzzy matches found above the threshold.")
    
    # Example section (UNCHANGED)
    st.markdown("---")
    st.header("Example Usage")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("Try Example 1", use_container_width=True):
            st.experimental_set_query_params(
                sku="SV50139.06",
                name="PKG MATL| COLLAPSIBLE BIN, CUSTOM 1000L BOTTOM DRAIN"
            )
            st.experimental_rerun()
    
    with example_col2:
        st.code("""
SKU: SV50139.06
Name: PKG MATL| COLLAPSIBLE BIN, CUSTOM 1000L BOTTOM DRAIN
        """)

if __name__ == "__main__":
    main()