import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# ===================== CORE DATA FUNCTIONS =====================

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess training data"""
    try:
        df = pd.read_excel('Training_Set.xlsx', engine='openpyxl')
        
        # Clean data - remove empty rows
        df = df[(df['sku number'].notna()) & (df['sku number'] != '') & 
                (df['sku name'].notna()) & (df['sku name'] != '')]
        
        # Filter valid product lines
        valid_product_lines = [
            'BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 
            'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 
            'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 
            'BioproductionContainments', 'BottleAssemblies',
            'ProductionCellCulture', 'RigidOther', 'SUDOther'
        ]
        df = df[df['cmr product line'].isin(valid_product_lines)]
        
        # Remove duplicates and clean SKU numbers
        df = df.drop_duplicates(subset=['sku number', 'sku name'])
        df['sku number'] = df['sku number'].str.replace(r'(INT_FINESS.*|BPD.*)', '', regex=True)
        
        # Apply volume-based CMR product line correction during data loading
        df['cmr product line'] = df.apply(lambda row: determine_correct_cmr_by_volume(
            row['sku name'], row['cmr product line']), axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_business_rules_data():
    """Load business rule book data"""
    try:
        return pd.read_excel('Business_Rule.xlsx', engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading Business Rule Book: {str(e)}")
        return None

@st.cache_resource
def create_similarity_index(df):
    """Create TF-IDF similarity index"""
    combined_text = (df['sku number'].astype(str) + " " + 
                    df['sku name'].astype(str)).str.lower()
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=3000,
        analyzer='char_wb',
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return vectorizer, tfidf_matrix

# ===================== VOLUME EXTRACTION & 2D/3D MAPPING =====================

def extract_volume_enhanced(sku_name):
    """Enhanced volume extraction with comprehensive pattern matching"""
    if not sku_name:
        return None
    
    sku_name = str(sku_name).upper().strip()
    
    # Liter patterns
    liter_patterns = [
        r'(\d+(?:\.\d+)?)\s*L(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*LITER',
        r'(\d+(?:\.\d+)?)\s*LITRE'
    ]
    
    for pattern in liter_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1))
    
    # Milliliter patterns
    ml_patterns = [
        r'(\d+(?:\.\d+)?)\s*M?ML(?:\s|$|[^\w])',
        r'(\d+(?:\.\d+)?)\s*MILLILITER',
        r'(\d+(?:\.\d+)?)\s*MILLILITRE'
    ]
    
    for pattern in ml_patterns:
        match = re.search(pattern, sku_name)
        if match:
            return float(match.group(1)) / 1000  # Convert to liters
    
    return None

def determine_correct_cmr_by_volume(sku_name, original_cmr_line):
    """Determine correct CMR product line based on 20L volume rule"""
    if original_cmr_line not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return original_cmr_line
    
    volume_l = extract_volume_enhanced(sku_name)
    if volume_l is None:
        return original_cmr_line
    
    return '2DBioProcessContainers' if volume_l <= 20 else '3DBioProcessContainers'

def create_2d_to_3d_mapping():
    """
    Create mapping from 2D product line codes to appropriate 3D codes.
    Based on analysis of the data structure.
    """
    return {
        '2JE': '2MH',  # GENERAL 2D -> PRODUCTAINER BPC
        '2JC': '2MH',  # LABTAINER -> PRODUCTAINER BPC
        '2PQ': '2PS',  # 2DBioProcessContainers Tieout -> 3DBioProcessContainers Tieout
        '2MD': '2MN',  # Map to 3D Manifold
        '2JD': '2MH',  # Map to PRODUCTAINER BPC
        '0CF': '0D8',  # 2D SINGLE -> SINGLE
        '2MB': '2MH',  # Map to PRODUCTAINER BPC
        '2MF': '2MJ',  # 2D TANK LINER -> 3D TANK LINERS
        '0D0': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z3U': '2MH',  # Map to PRODUCTAINER BPC
        'Z6R': '0D8',  # 2D SINGLE -> SINGLE
        '0CZ': '0D8',  # 2D SINGLE -> SINGLE
        'Z3R': '0D8',  # 2D SINGLE -> SINGLE
        'Z2K': '2MN',  # MANIFOLD -> 3D MANIFOLD
        'Z37': '0D8'   # 2D SINGLE -> SINGLE
    }

def get_2d_to_3d_mapping():
    # Duplicate of create_2d_to_3d_mapping, remove to avoid confusion
    pass

def get_3d_to_2d_mapping():
    """3D to 2D product line code mapping"""
    return {
        '2MH': '2JE', '2MJ': '2MF', '2MO': '2JE', '2PS': '2JE', '2MN': '2MD',
        'Z2H': '2JE', '2ML': '2JE', 'Z39': '2JE', 'Z6M': '2JE', '2MM': '2JE',
        '0D8': '2MD', '0EG': '2JE', '3D6': '2JE', '3WO': '2JE', 'Z3Q': '2JE',
        '262': '2JE', '2MG': '2JE'
    }

def get_product_line_name(product_line_code, is_2d=True):
    """Get appropriate product line name"""
    if is_2d:
        mapping = {
            '2JE': 'FLEXIBLE CONSUMABLES 2D', '2JC': 'LABTAINER', '2JD': 'GENERAL 2D',
            '2PQ': '2DBioProcessContainers Tieout', '2MD': '2D MANIFOLD', '0CF': '2D SINGLE',
            '2MB': '2D HARVESTAINER', '2MF': '2D TANK LINER', '0D0': '2D MANIFOLD',
            'Z3U': 'MANIFOLD', 'Z6R': '2D SINGLE', '0CZ': '2D MANIFOLD', 'Z3R': '2D SINGLE',
            'Z2K': 'MANIFOLD', 'Z37': 'FLEXIBLE CONSUMABLES 2D'
        }
        return mapping.get(product_line_code, 'FLEXIBLE CONSUMABLES 2D')
    else:
        mapping = {
            '2MH': 'PRODUCTAINER BPC', '2MJ': '3D TANK LINERS', '2MO': '3D PRODUCTAINER',
            '2PS': '3D PRODUCTAINER', '2MN': '3D MANIFOLD', 'Z2H': '3D PRODUCTAINER',
            '2ML': '3D PRODUCTAINER', 'Z39': '3D PRODUCTAINER', 'Z6M': '3D PRODUCTAINER',
            '2MM': 'OTHER OUTER SUPPORT CONTAINERS', '0D8': '3D MANIFOLD', '0EG': '3D PRODUCTAINER',
            '3D6': '3D PRODUCTAINER', '3WO': '3D PRODUCTAINER', 'Z3Q': '3D PRODUCTAINER',
            '262': '3D PRODUCTAINER', '2MG': '3D PRODUCTAINER'
        }
        return mapping.get(product_line_code, 'PRODUCTAINER BPC')

def adjust_product_line_for_volume(original_cmr, product_line_code, product_line_name, sku_name):
    """Adjust product line code and name based on volume-determined CMR classification"""
    if original_cmr not in ['2DBioProcessContainers', '3DBioProcessContainers']:
        return product_line_code, product_line_name, original_cmr

    correct_cmr = determine_correct_cmr_by_volume(sku_name, original_cmr)

    # Always map 2D code to 3D code if CMR is 3DBioProcessContainers
    if correct_cmr == '3DBioProcessContainers':
        mapped_code = create_2d_to_3d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=False)
        return mapped_code, mapped_name, correct_cmr
    # Always map 3D code to 2D code if CMR is 2DBioProcessContainers
    elif correct_cmr == '2DBioProcessContainers':
        mapped_code = get_3d_to_2d_mapping().get(product_line_code, product_line_code)
        mapped_name = get_product_line_name(mapped_code, is_2d=True)
        return mapped_code, mapped_name, correct_cmr
    # Fallback (should not hit)
    return product_line_code, product_line_name, correct_cmr

# ===================== SIMILARITY & PREDICTION FUNCTIONS =====================

@lru_cache(maxsize=500)
def calculate_simple_similarity(s1, s2):
    """Cached similarity calculation"""
    if not s1 or not s2:
        return 0.0
    
    s1, s2 = s1.lower(), s2.lower()
    if s1 in s2 or s2 in s1:
        return 85.0
    
    set1, set2 = set(s1), set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / union) * 100 if union > 0 else 0.0

def get_exact_predictions(df, sku_partial, name_partial):
    """Find exact substring matches"""
    if not sku_partial.strip() and not name_partial.strip():
        return pd.DataFrame()
    
    df_copy = df.copy()
    
    if sku_partial.strip():
        df_copy = df_copy[df_copy['sku number'].astype(str).str.upper().str.contains(
            sku_partial.upper(), na=False, regex=False)]
    
    if name_partial.strip():
        df_copy = df_copy[df_copy['sku name'].astype(str).str.lower().str.contains(
            name_partial.lower(), na=False, regex=False)]
    
    return df_copy.drop_duplicates(
        subset=['sku number', 'sku name', 'product line code', 'cmr product line']
    ).head(10)

def get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k=5):
    """Get fuzzy predictions with volume-based 2D/3D mapping"""
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    query_text = f"{sku_partial} {name_partial}".lower()
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = np.argsort(similarities)[-top_k*3:][::-1]
    results = []
    seen_combinations = set()
    
    for idx in top_indices:
        if len(results) >= top_k or similarities[idx] < 0.1:
            continue
            
        row = df.iloc[idx]
        combination_id = f"{row['product line code']}|{row['cmr product line']}"
        
        if combination_id not in seen_combinations:
            seen_combinations.add(combination_id)
            
            # Apply volume-based mapping using input name
            adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                row['cmr product line'], row['product line code'], 
                row['product line name'], name_partial
            )
            
            results.append({
                'sku_number': row['sku number'],
                'sku_name': row['sku name'],
                'product_line_code': adj_code,
                'cmr_product_line': correct_cmr,
                'product_line_name': adj_name,
                'sub_platform': row['sub platform'],
                'sku_score': round(calculate_simple_similarity(sku_partial, str(row['sku number'])), 2),
                'name_score': round(calculate_simple_similarity(name_partial, str(row['sku name'])), 2),
                'combined_score': round(similarities[idx] * 100, 2)
            })
    
    return results

def ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix):
    """Ultra-fast bulk prediction with optimized 2D/3D mapping"""
    # Prepare input data
    input_clean = input_df.copy()
    input_clean['sku number'] = input_clean['sku number'].fillna('').astype(str)
    input_clean['sku name'] = input_clean['sku name'].fillna('').astype(str)
    query_texts = (input_clean['sku number'] + ' ' + input_clean['sku name']).str.lower()
    
    # Vectorized similarity computation
    with st.spinner("Computing similarities..."):
        query_vectors = vectorizer.transform(query_texts)
        similarities = cosine_similarity(query_vectors, tfidf_matrix)
    
    # Find top 3 matches efficiently
    with st.spinner("Finding top matches..."):
        top_k = 3
        top_indices = np.argpartition(-similarities, range(top_k), axis=1)[:, :top_k]
        sorted_indices = np.argsort(-similarities[np.arange(similarities.shape[0])[:, None], top_indices], axis=1)
        final_indices = top_indices[np.arange(similarities.shape[0])[:, None], sorted_indices]
        top_scores = similarities[np.arange(similarities.shape[0])[:, None], final_indices]
    
    # Build results with volume-based mapping
    with st.spinner("Building results..."):
        results = []
        for i, (_, row) in enumerate(input_df.iterrows()):
            result_row = row.copy()
            
            for pred_num in range(3):
                if pred_num < final_indices.shape[1] and top_scores[i, pred_num] >= 0.1:
                    match_row = df.iloc[final_indices[i, pred_num]]
                    
                    # Apply volume-based mapping
                    adj_code, adj_name, correct_cmr = adjust_product_line_for_volume(
                        match_row['cmr product line'], match_row['product line code'],
                        match_row['product line name'], row.get('sku name', '')
                    )
                    
                    prefix = f'Prediction {pred_num+1}: '
                    result_row[f'{prefix}SKU Number'] = match_row['sku number']
                    result_row[f'{prefix}SKU Name'] = match_row['sku name']
                    result_row[f'{prefix}CMR Product Line'] = correct_cmr
                    result_row[f'{prefix}Product Line Name'] = adj_name
                    result_row[f'{prefix}Product Line Code'] = adj_code
                    result_row[f'{prefix}Business Unit'] = match_row['sub platform']
                    result_row[f'{prefix}Confidence Score'] = round(top_scores[i, pred_num] * 100, 2)
                else:
                    prefix = f'Prediction {pred_num+1}: '
                    for suffix in ['SKU Number', 'SKU Name', 'CMR Product Line', 
                                 'Product Line Name', 'Product Line Code', 'Business Unit']:
                        result_row[f'{prefix}{suffix}'] = 'No Match Found'
                    result_row[f'{prefix}Confidence Score'] = 0.0
            
            results.append(result_row)
    
    return pd.DataFrame(results)

# ===================== BUSINESS RULES =====================

def get_business_rule(product_line_code, df_rules):
    """Get business rule for product line code"""
    if df_rules is None:
        return None, None
    
    try:
        rule_row = df_rules[df_rules['product line code'] == product_line_code]
        if not rule_row.empty:
            rule = rule_row.iloc[0]
            return (rule.get('Top Trigrams in SKU Name', 'N/A'),
                   rule.get('Top Prefixes in SKU No.', 'N/A'))
        return None, None
    except Exception as e:
        st.error(f"Error retrieving business rule: {str(e)}")
        return None, None

# ===================== DISPLAY FUNCTIONS =====================

def safe_dataframe_display(df, key=None):
    """Safe dataframe display with fallback"""
    try:
        st.dataframe(df, key=key)
    except Exception:
        st.warning("Display issue detected. Showing alternative preview.")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        if len(df) > 0:
            st.json(df.head(3).to_dict('records'))

def display_exact_matches(exact_matches):
    """Display exact match results"""
    st.subheader("✅ Exact Matches Found")
    st.success(f"{len(exact_matches)} Strong Prediction(s) Found")
    
    for idx, row in exact_matches.iterrows():
        # Adjust code/name for 2D/3D CMR
        adj_code, adj_name, adj_cmr = adjust_product_line_for_volume(
            row['cmr product line'], row['product line code'], row['product line name'], row['sku name']
        )
        with st.expander(f"Match {idx + 1}: {adj_code} - {adj_cmr}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**SKU Number:**", row['sku number'])
                st.write("**Product Line Code:**", adj_code)
                st.write("**Product Line Name:**", adj_name)
            with col2:
                st.write("**SKU Name:**", row['sku name'])
                st.write("**CMR Product Line:**", adj_cmr)
                st.write("**Business Unit:**", row['sub platform'])

def display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules):
    """Display fuzzy match results"""
    if not fuzzy_matches:
        st.warning("No fuzzy matches found above the threshold.")
        return
    
    st.markdown("### 📥 Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Input SKU Number:** {sku_input if sku_input else 'N/A'}")
    with col2:
        st.info(f"**Input SKU Name:** {name_input if name_input else 'N/A'}")
    
    st.markdown("---")
    st.subheader("🔍 Top Fuzzy Predictions")
    
    for i, match in enumerate(fuzzy_matches, 1):
        confidence_color = "🟢" if match['combined_score'] >= 80 else "🟡" if match['combined_score'] >= 60 else "🔴"
        
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
                truncated_name = match['sku_name'][:50] + "..." if len(match['sku_name']) > 50 else match['sku_name']
                st.write("**SKU Name:**", truncated_name)
                st.write("**CMR Product Line:**", match['cmr_product_line'])
                st.write("**Business Unit:**", match['sub_platform'])
            
            with col3:
                st.metric("Confidence Score", f"{match['combined_score']}%")
            
            if len(match['sku_name']) > 50:
                st.write("**Full SKU Name:**", match['sku_name'])

            # Business Rule Section
            st.markdown("---")
            st.markdown("**📋 Business Rule Identified:**")
            
            sku_name_pattern, sku_prefix_pattern = get_business_rule(match['product_line_code'], df_rules)
            
            if sku_name_pattern and sku_prefix_pattern:
                st.info(f"**Common SKU Name Pattern:** {sku_name_pattern}")
                st.info(f"**Prefix Pattern Found:** {sku_prefix_pattern}")
            else:
                st.warning("No business rule found for this product line code")

# ===================== UI TABS =====================

def single_sku_tab(df, df_rules, vectorizer, tfidf_matrix):
    """Single SKU processing tab"""
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
                placeholder="e.g., PKG MATL| COLLAPSIBLE BIN",
                help="Enter the SKU name (partial matches allowed)"
            )
        
        submitted = st.form_submit_button("🔍 Classify SKU", type="primary", use_container_width=True)
    
    if submitted:
        if not sku_input.strip() and not name_input.strip():
            st.warning("Please enter at least one field (SKU Number or SKU Name)")
            return
        
        with st.spinner("Analyzing SKU patterns..."):
            exact_matches = get_exact_predictions(df, sku_input, name_input)
            fuzzy_matches = get_fuzzy_predictions(df, sku_input, name_input, vectorizer, tfidf_matrix, top_k=8)
        
        st.markdown("---")
        st.header("Classification Results")
        
        if not exact_matches.empty:
            display_exact_matches(exact_matches)
        else:
            st.info("No exact matches found. Showing fuzzy predictions below.")
        
        display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules)

def bulk_processing_tab(df, vectorizer, tfidf_matrix):
    """Optimized bulk SKU processing tab"""
    st.header("🚀 Bulk SKU Classification")
    st.info("Upload a CSV or Excel file with 'sku number' and 'sku name' columns")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain columns: sku name, sku number"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                input_df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                input_df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
            
            st.success(f"File uploaded successfully! Found {len(input_df)} rows.")
            
            # Display preview
            with st.expander("📋 File Preview", expanded=False):
                safe_dataframe_display(input_df.head(10), key="file_preview")
            
            # Check required columns
            required_cols = ['sku number', 'sku name']
            input_columns_lower = [col.lower() for col in input_df.columns]
            missing_cols = [col for col in required_cols if col.lower() not in input_columns_lower]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info("Please ensure your file contains 'sku number' and 'sku name' columns")
                return
            
            # Normalize column names
            column_mapping = {}
            for col in input_df.columns:
                if col.lower() == 'sku number':
                    column_mapping[col] = 'sku number'
                elif col.lower() == 'sku name':
                    column_mapping[col] = 'sku name'
            
            input_df = input_df.rename(columns=column_mapping)
            
            # Show processing info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows to Process", len(input_df))
            with col2:
                est_time = len(input_df) * 0.01
                st.metric("Estimated Time", f"{est_time:.1f}s")
            
            # Process button
            if st.button("🚀 Process Classifications", type="primary", use_container_width=True):
                start_time = pd.Timestamp.now()
                
                # Process data
                result_df = ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix)
                
                # Apply filters
                initial_rows = len(result_df)
                filter_patterns = ['unknown', 'ADJCST', 'FADJ']
                
                for pattern in filter_patterns:
                    result_df = result_df[~result_df['sku name'].str.contains(pattern, case=False, na=False)]
                    result_df = result_df[~result_df['sku number'].str.contains(pattern, case=False, na=False)]
                
                filtered_rows = initial_rows - len(result_df)
                result_df = result_df.reset_index(drop=True)
                
                end_time = pd.Timestamp.now()
                processing_time = (end_time - start_time).total_seconds()
                
                st.success(f"Processing completed in {processing_time:.2f} seconds!")
                
                if filtered_rows > 0:
                    st.info(f"🔍 Filtered out {filtered_rows} rows containing excluded patterns")
                
                # Performance metrics
                st.subheader("📊 Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col2:
                    rows_per_sec = len(input_df) / processing_time if processing_time > 0 else 0
                    st.metric("Rows/Second", f"{rows_per_sec:.1f}")
                with col3:
                    successful_matches = len(result_df[result_df['Prediction 1: SKU Number'] != 'No Match Found'])
                    st.metric("Successful Matches", successful_matches)
                with col4:
                    success_rate = (successful_matches / len(result_df) * 100) if len(result_df) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Results preview
                st.subheader("🔍 Results Preview")
                prediction_cols = [col for col in result_df.columns if col.startswith('Prediction 1:')]
                preview_cols = ['sku number', 'sku name'] + prediction_cols
                safe_dataframe_display(result_df[preview_cols].head(10), key="results_preview")
                
                # Download section
                st.subheader("💾 Download Results")
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"SKU_Classifications_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ===================== MAIN APPLICATION =====================

def main():
    st.set_page_config(
        page_title="SKU Classification System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            st.image("logo2.png", width=1000)
        except:
            st.write("🤖")
    
    with col2:
        st.title("SKU Product Line Classification System")
    
    st.markdown("---")
    
    # Initialize data
    if 'df' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            if st.session_state.df is not None:
                st.session_state.vectorizer, st.session_state.tfidf_matrix = create_similarity_index(st.session_state.df)
    
    if 'df_rules' not in st.session_state:
        st.session_state.df_rules = load_business_rules_data()

    df = st.session_state.df
    df_rules = st.session_state.df_rules
    
    if df is None:
        st.error("Failed to load data. Please check if the Training_Set.xlsx file exists.")
        return
    
    st.success(f'✅ Model ready! Trained on {len(df)} SKU patterns')

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Single SKU Classification", "📂 Bulk Classification"])

    # Single SKU tab
    with tab1:
        single_sku_tab(df, df_rules, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    # Bulk processing tab
    with tab2:
        bulk_processing_tab(df, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        st.metric("Training SKUs", len(df))
        st.metric("Product Line Codes", 128)
        st.metric("CMR Product Lines", df['cmr product line'].nunique())

        st.header("🚀 How it works")
        st.markdown("""
        1. **Volume Analysis**: Extracts volume from SKU names
        2. **2D/3D Classification**: ≤20L=2D, >20L=3D  
        3. **Exact Match**: Finds substring matches
        4. **Fuzzy Search**: AI-powered similarity matching
        5. **Code Mapping**: Auto-adjusts product line codes
        """)

        st.header("📋 Volume Rules")
        st.info("""
        **2D BioProcess Containers**: ≤ 20 Liters
        **3D BioProcess Containers**: > 20 Liters
        
        System automatically maps product line codes based on extracted volume.
        """)

        # Business rules download
        try:
            with open('Business_Rule.xlsx', 'rb') as file:
                business_rule_data = file.read()
                st.download_button(
                    label="📥 Download Business Rules",
                    data=business_rule_data,
                    file_name="Business_Rule.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        except:
            pass

if __name__ == "__main__":
    main()