import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import warnings
import io
warnings.filterwarnings('ignore')

# Fix for PyArrow DLL issues in some environments
import os
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Alternative dataframe display function to handle PyArrow issues
def safe_dataframe_display(df, key=None):
    """Display dataframe with fallback methods for PyArrow issues"""
    try:
        st.dataframe(df, key=key)
    except Exception as e:
        st.warning("PyArrow display issue detected. Showing fallback preview below.")
        try:
            # Try to show as markdown table (no PyArrow dependency)
            st.markdown(df.head(10).to_markdown(index=False), unsafe_allow_html=True)
        except Exception:
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {list(df.columns)}")
            if len(df) > 0:
                st.json(df.head(3).to_dict('records'))

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_excel('Training_Set.xlsx')
        
        # Clean data
        df = df[(df['sku number'].notna()) & (df['sku number'] != '') & 
                (df['sku name'].notna()) & (df['sku name'] != '')]
        
        # Filter valid product lines
        valid_product_lines = ['BEAService', 'BEAHardware', 'BEAOther', 'HardwareConsumables', 
                              'SUTAutomation','2DBioProcessContainers', '3DBioProcessContainers', 
                              'FillFinish', 'FlexibleOther','FluidTransferAssemblies', 
                              'BioproductionContainments', 'BottleAssemblies',
                              'ProductionCellCulture', 'RigidOther', 'SUDOther']
        df = df[df['cmr product line'].isin(valid_product_lines)]
        
        # Remove duplicates based on sku number and sku name
        df = df.drop_duplicates(subset=['sku number', 'sku name'])
        
        # Clean SKU numbers
        df['sku number'] = df['sku number'].str.replace(r'(INT_FINESS.*|BPD.*)', '', regex=True)
        
        # Update CMR product line based on volume
        def extract_volume(sku_name):
            match = re.search(r'(\d+)\s*L', sku_name.upper())
            return int(match.group(1)) if match else None
        
        df['volume_l'] = df['sku name'].apply(extract_volume)
        
        def update_cmr_product_line(row):
            if row['cmr product line'] in ['2DBioProcessContainers', '3DBioProcessContainers']:
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
    file_path = 'Business_Rule.xlsx'
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading Business Rule Book: {str(e)}")
        return None

@st.cache_data
def load_business_rules_data():
    """Load the business rule book data for rule lookup"""
    file_path = 'Business_Rule.xlsx'
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error loading Business Rule Book data: {str(e)}")
        return None

@st.cache_resource
def create_similarity_index(df):
    """Create TF-IDF similarity index for fast fuzzy matching"""
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

def get_exact_predictions(df, sku_partial, name_partial):
    """Simple exact/substring matches"""
    if not sku_partial.strip() and not name_partial.strip():
        return pd.DataFrame()
    
    df_copy = df.copy()
    
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
    
    exact_matches = df_copy.drop_duplicates(
        subset=['sku number', 'sku name', 'product line code', 'cmr product line']
    )
    
    return exact_matches.head(10)

def get_fuzzy_predictions(df, sku_partial, name_partial, vectorizer, tfidf_matrix, top_k=5):
    """Optimized fuzzy matching using TF-IDF cosine similarity"""
    if not sku_partial.strip() and not name_partial.strip():
        return []
    
    query_text = f"{sku_partial} {name_partial}".lower()
    query_vector = vectorizer.transform([query_text])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k*3:][::-1]
    
    results = []
    seen_combinations = set()
    
    for idx in top_indices:
        if len(results) >= top_k:
            break
            
        if similarities[idx] < 0.1:
            continue
            
        row = df.iloc[idx]
        combination_id = f"{row['product line code']}|{row['cmr product line']}"
        
        if combination_id not in seen_combinations:
            seen_combinations.add(combination_id)
            
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

def get_business_rule(product_line_code, df_rules):
    """Get business rule for a specific product line code"""
    if df_rules is None:
        return None, None
    
    try:
        rule_row = df_rules[df_rules['product line code'] == product_line_code]
        
        if not rule_row.empty:
            rule = rule_row.iloc[0]
            sku_name_pattern = rule.get('Top Trigrams in SKU Name', 'N/A')
            sku_prefix_pattern = rule.get('Top Prefixes in SKU No.', 'N/A')
            return sku_name_pattern, sku_prefix_pattern
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error retrieving business rule: {str(e)}")
        return None, None

# OPTIMIZED BULK PROCESSING FUNCTIONS
def vectorized_bulk_predictions(input_df, df, vectorizer, tfidf_matrix, top_k=1, batch_size=100):
    """
    Highly optimized bulk prediction using vectorized operations and batching
    """
    results = []
    total_rows = len(input_df)
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in batches to manage memory
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = input_df.iloc[batch_start:batch_end].copy()
        
        # Update progress
        progress = batch_end / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing batch {batch_start//batch_size + 1}/{(total_rows-1)//batch_size + 1} ({batch_end}/{total_rows} SKUs)')
        
        # Vectorized text preparation
        batch_df['sku number'] = batch_df['sku number'].fillna('').astype(str)
        batch_df['sku name'] = batch_df['sku name'].fillna('').astype(str)
        
        query_texts = (batch_df['sku number'] + ' ' + batch_df['sku name']).str.lower()
        
        # Vectorized similarity computation
        query_vectors = vectorizer.transform(query_texts)
        similarities = cosine_similarity(query_vectors, tfidf_matrix)
        
        # Find best matches for entire batch
        best_indices = np.argmax(similarities, axis=1)
        best_scores = np.max(similarities, axis=1)
        
        # Build results for this batch
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            result_row = row.copy()
            
            best_idx = best_indices[i]
            best_score = best_scores[i]
            
            if best_score >= 0.1:  # Minimum similarity threshold
                match_row = df.iloc[best_idx]
                result_row['Prediction 1: SKU Number'] = match_row['sku number']
                result_row['Prediction 1: SKU Name'] = match_row['sku name']
                result_row['Prediction 1: CMR Product Line'] = match_row['cmr product line']
                result_row['Prediction 1: Product Line Name'] = match_row['product line name']
                result_row['Prediction 1: Product Line Code'] = match_row['product line code']
                result_row['Prediction 1: Business Unit'] = match_row['sub platform']
                result_row['Prediction 1: Confidence Score'] = round(best_score * 100, 2)
            else:
                result_row['Prediction 1: SKU Number'] = 'No Match Found'
                result_row['Prediction 1: SKU Name'] = 'No Match Found'
                result_row['Prediction 1: CMR Product Line'] = 'No Match Found'
                result_row['Prediction 1: Product Line Name'] = 'No Match Found'
                result_row['Prediction 1: Product Line Code'] = 'No Match Found'
                result_row['Prediction 1: Business Unit'] = 'No Match Found'
                result_row['Prediction 1: Confidence Score'] = 0.0
            
            results.append(result_row)
    
    progress_bar.progress(1.0)
    status_text.text('Processing complete!')
    
    return pd.DataFrame(results)

def ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix):
    """
    Ultra-fast bulk prediction using pure NumPy operations
    """
    # Prepare all query texts at once
    input_df_clean = input_df.copy()
    input_df_clean['sku number'] = input_df_clean['sku number'].fillna('').astype(str)
    input_df_clean['sku name'] = input_df_clean['sku name'].fillna('').astype(str)
    
    query_texts = (input_df_clean['sku number'] + ' ' + input_df_clean['sku name']).str.lower()
    
    # Single vectorization call for all queries
    with st.spinner("Vectorizing queries..."):
        query_vectors = vectorizer.transform(query_texts)
    
    # Single similarity computation for all queries
    with st.spinner("Computing similarities..."):
        similarities = cosine_similarity(query_vectors, tfidf_matrix)
    
    # Find best matches using vectorized operations
    with st.spinner("Finding best matches..."):
        best_indices = np.argmax(similarities, axis=1)
        best_scores = np.max(similarities, axis=1)
    
    # Build results using vectorized operations where possible
    with st.spinner("Building results..."):
        results = []
        
        # Pre-extract all match data
        match_data = df.iloc[best_indices][['sku number', 'sku name', 'cmr product line', 
                                          'product line name', 'product line code', 'sub platform']].values
        
        for i, (idx, row) in enumerate(input_df.iterrows()):
            result_row = row.copy()
            
            if best_scores[i] >= 0.1:
                match_row_data = match_data[i]
                result_row['Prediction 1: SKU Number'] = match_row_data[0]
                result_row['Prediction 1: SKU Name'] = match_row_data[1]
                result_row['Prediction 1: CMR Product Line'] = match_row_data[2]
                result_row['Prediction 1: Product Line Name'] = match_row_data[3]
                result_row['Prediction 1: Product Line Code'] = match_row_data[4]
                result_row['Prediction 1: Business Unit'] = match_row_data[5]
                result_row['Prediction 1: Confidence Score'] = round(best_scores[i] * 100, 2)
            else:
                # No match found
                no_match_values = ['No Match Found'] * 6 + [0.0]
                prediction_keys = ['Prediction 1: SKU Number', 'Prediction 1: SKU Name', 
                                 'Prediction 1: CMR Product Line', 'Prediction 1: Product Line Name',
                                 'Prediction 1: Product Line Code', 'Prediction 1: Business Unit',
                                 'Prediction 1: Confidence Score']
                
                for key, value in zip(prediction_keys, no_match_values):
                    result_row[key] = value
            
            results.append(result_row)
    
    return pd.DataFrame(results)

def process_bulk_predictions(input_df, df, vectorizer, tfidf_matrix):
    """Original process bulk predictions for comparison"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(input_df)
    
    for idx, row in input_df.iterrows():
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f'Processing {idx + 1} of {total_rows} SKUs...')
        
        sku_number = str(row.get('sku number', '')) if pd.notna(row.get('sku number')) else ''
        sku_name = str(row.get('sku name', '')) if pd.notna(row.get('sku name')) else ''
        
        # Get top fuzzy prediction
        fuzzy_matches = get_fuzzy_predictions(
            df, sku_number, sku_name, vectorizer, tfidf_matrix, top_k=1
        )
        
        # Prepare result row
        result_row = row.copy()  # Keep all original columns
        
        if fuzzy_matches:
            top_match = fuzzy_matches[0]
            result_row['Prediction 1: SKU Number'] = top_match['sku_number']
            result_row['Prediction 1: SKU Name'] = top_match['sku_name']
            result_row['Prediction 1: CMR Product Line'] = top_match['cmr_product_line']
            result_row['Prediction 1: Product Line Name'] = top_match['product_line_name']
            result_row['Prediction 1: Product Line Code'] = top_match['product_line_code']
            result_row['Prediction 1: Business Unit'] = top_match['sub_platform']
            result_row['Prediction 1: Confidence Score'] = top_match['combined_score']
        else:
            result_row['Prediction 1: SKU Number'] = 'No Match Found'
            result_row['Prediction 1: SKU Name'] = 'No Match Found'
            result_row['Prediction 1: CMR Product Line'] = 'No Match Found'
            result_row['Prediction 1: Product Line Name'] = 'No Match Found'
            result_row['Prediction 1: Product Line Code'] = 'No Match Found'
            result_row['Prediction 1: Business Unit'] = 'No Match Found'
            result_row['Prediction 1: Confidence Score'] = 0.0
        
        results.append(result_row)
    
    progress_bar.progress(1.0)
    status_text.text('Processing complete!')
    
    return pd.DataFrame(results)

def display_exact_matches(exact_matches):
    """Display exact match results"""
    st.subheader("✅ Exact Matches Found")
    st.success(f"{len(exact_matches)} Strong Prediction(s) Found")
    
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

def display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules):
    """Display fuzzy match results"""
    if not fuzzy_matches:
        st.warning("No fuzzy matches found above the threshold.")
        return
    
    st.markdown("### 📥 Input Parameters")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        st.info(f"**Input SKU Number:** {sku_input if sku_input else 'N/A'}")
    with input_col2:
        st.info(f"**Input SKU Name:** {name_input if name_input else 'N/A'}")
    
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
    
    # Process form submission
    if submitted:
        if not sku_input.strip() and not name_input.strip():
            st.warning("Please enter at least one field (SKU Number or SKU Name)")
            return
        
        with st.spinner("Analyzing SKU patterns..."):
            exact_matches = get_exact_predictions(df, sku_input, name_input)
            fuzzy_matches = get_fuzzy_predictions(
                df, sku_input, name_input, vectorizer, tfidf_matrix, top_k=8
            )
        
        st.markdown("---")
        st.header("Classification Results")
        
        # Display results
        if not exact_matches.empty:
            display_exact_matches(exact_matches)
        else:
            st.info("No exact matches found. Showing fuzzy predictions below.")
        
        display_fuzzy_matches(fuzzy_matches, sku_input, name_input, df_rules)
    
    # Example section
    st.markdown("---")
    st.header("Example Usage")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("Try Example", use_container_width=True):
            st.rerun()
    
    with example_col2:
        st.code("""
SKU: SV50139.06
Name: PKG MATL| COLLAPSIBLE BIN, CUSTOM 1000L BOTTOM DRAIN
        """)

def optimized_bulk_processing_tab(df, vectorizer, tfidf_matrix):
    """
    Optimized bulk SKU processing tab with support for both Excel and CSV files
    """
    st.header("🚀 Optimized Bulk SKU Classification")
    st.info("Upload a CSV or Excel file with 'sku number' and 'sku name' columns for ultra-fast bulk processing")
    
    # Processing method selection
    processing_method = st.selectbox(
        "Select Processing Method",
        options=["Ultra Fast (Recommended)", "Batch Processing", "Standard"],
        help="Ultra Fast: Best for large files (1000+ rows)\nBatch Processing: Good for medium files (100-1000 rows)\nStandard: Original method"
    )
    
    # Batch size configuration for batch processing
    if processing_method == "Batch Processing":
        batch_size = st.slider("Batch Size", min_value=50, max_value=500, value=100, step=50,
                              help="Larger batch sizes are faster but use more memory")
    
    # File upload - supporting both CSV and Excel
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain columns: sku name, sku number, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                input_df = pd.read_csv(uploaded_file)
                st.success(f"CSV file uploaded successfully! Found {len(input_df)} rows.")
            elif file_extension in ['xlsx', 'xls']:
                input_df = pd.read_excel(uploaded_file)
                st.success(f"Excel file uploaded successfully! Found {len(input_df)} rows.")
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
            
            # Display file preview (using safe display method)
            with st.expander("📋 File Preview", expanded=False):
                safe_dataframe_display(input_df.head(10), key="file_preview")
            
            # Check required columns (case-insensitive)
            required_cols = ['sku number', 'sku name']
            input_columns_lower = [col.lower() for col in input_df.columns]
            missing_cols = []
            
            for req_col in required_cols:
                if req_col.lower() not in input_columns_lower:
                    missing_cols.append(req_col)
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info("Please ensure your file contains 'sku number' and 'sku name' columns (case-insensitive)")
                st.info(f"Available columns: {list(input_df.columns)}")
                return
            
            # Normalize column names to match expected format
            column_mapping = {}
            for col in input_df.columns:
                if col.lower() == 'sku number':
                    column_mapping[col] = 'sku number'
                elif col.lower() == 'sku name':
                    column_mapping[col] = 'sku name'
            
            input_df = input_df.rename(columns=column_mapping)
            
            # Show processing information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows to Process", len(input_df))
            with col2:
                est_time = len(input_df) * 0.01 if processing_method == "Ultra Fast (Recommended)" else len(input_df) * 0.1
                st.metric("Estimated Time", f"{est_time:.1f}s")
            with col3:
                st.metric("Method", processing_method.split(' ')[0])
            
            # Process button
            if st.button("🚀 Process Bulk Classifications", type="primary", use_container_width=True):
                
                start_time = pd.Timestamp.now()
                
                # Choose processing method
                if processing_method == "Ultra Fast (Recommended)":
                    result_df = ultra_fast_bulk_predictions(input_df, df, vectorizer, tfidf_matrix)
                elif processing_method == "Batch Processing":
                    result_df = vectorized_bulk_predictions(input_df, df, vectorizer, tfidf_matrix, batch_size=batch_size)
                else:
                    # Use original method
                    result_df = process_bulk_predictions(input_df, df, vectorizer, tfidf_matrix)
                
                end_time = pd.Timestamp.now()
                processing_time = (end_time - start_time).total_seconds()
                
                st.success(f"Bulk processing completed in {processing_time:.2f} seconds!")
                
                # Display performance metrics
                st.subheader("📊 Performance Metrics")
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                with perf_col2:
                    rows_per_sec = len(input_df) / processing_time if processing_time > 0 else 0
                    st.metric("Rows/Second", f"{rows_per_sec:.1f}")
                
                with perf_col3:
                    successful_matches = len(result_df[result_df['Prediction 1: SKU Number'] != 'No Match Found'])
                    st.metric("Successful Matches", successful_matches)
                
                with perf_col4:
                    success_rate = (successful_matches / len(result_df) * 100) if len(result_df) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Display results preview (using safe display method)
                st.subheader("🔍 Results Preview")
                prediction_cols = [col for col in result_df.columns if col.startswith('Prediction 1:')]
                preview_cols = ['sku number', 'sku name'] + prediction_cols
                safe_dataframe_display(result_df[preview_cols].head(10), key="results_preview")
                
                # Download results section
                st.subheader("💾 Download Results")
                
                # Create timestamp for filename
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                
                # Single CSV download (more reliable)
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"SKU_Classifications_Results_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Additional file format options
                st.markdown("---")
                st.subheader("📋 Alternative Export Options")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Tab-separated values (can be opened in Excel)
                    tsv_data = result_df.to_csv(index=False, sep='\t').encode('utf-8')
                    st.download_button(
                        label="📊 Download as TSV (Excel Compatible)",
                        data=tsv_data,
                        file_name=f"SKU_Classifications_Results_{timestamp}.tsv",
                        mime="text/tab-separated-values",
                        use_container_width=True,
                        help="Tab-separated file that opens nicely in Excel"
                    )
                
                with export_col2:
                    # Pipe-separated values (alternative format)
                    psv_data = result_df.to_csv(index=False, sep='|').encode('utf-8')
                    st.download_button(
                        label="📝 Download as PSV",
                        data=psv_data,
                        file_name=f"SKU_Classifications_Results_{timestamp}.psv",
                        mime="text/plain",
                        use_container_width=True,
                        help="Pipe-separated format for systems that need different delimiters"
                    )
                
                # Instructions for Excel users
                st.info("💡 **For Excel users**: Download the TSV file - it will open directly in Excel with proper column separation.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid CSV or Excel file with the correct columns.")
            import traceback
            st.code(traceback.format_exc())

def main():
    st.set_page_config(
        page_title="SKU Classification Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            col1.image("logo2.png", width=1000)
        except:
            st.write("🤖")  # Fallback if logo not found
    
    with col2:
        st.title("SKU Product Line Classification Chatbot")
    
    st.markdown("---")
    
    # Initialize data in session state
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
    
    st.success(f'Model ready! Trained on {len(df)} SKU patterns')

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Single SKU Classification", "📂 Bulk Classification"])

    # --- Tab 1: Single SKU ---
    with tab1:
        single_sku_tab(df, df_rules, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    # --- Tab 2: Bulk Upload ---
    with tab2:
        optimized_bulk_processing_tab(df, st.session_state.vectorizer, st.session_state.tfidf_matrix)

    # Sidebar
    with st.sidebar:
        st.header("📊 SKU Information")
        st.metric("Unique SKUs", len(df))
        st.metric("Product Line Codes", 128)
        st.metric("CMR Product Lines", df['cmr product line'].nunique())

        st.header("🚀 How it works")
        st.markdown("""
        1. **Exact Match**: Finds exact substring matches  
        2. **Fuzzy Search**: Uses intelligent similarity matching  
        3. **No Duplicates**: Ensures unique predictions  
        4. **Optimized Processing**: Ultra-fast bulk operations
        """)

        st.header("📋 Resources")
        business_rule_data = load_business_rule_book()
        if business_rule_data:
            st.download_button(
                label="📥 Download Business Rule",
                data=business_rule_data,
                file_name="Business_Rule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    

if __name__ == "__main__":
    main()