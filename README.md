# SKU Product Line Classification Chatbot

This project is a Streamlit-based chatbot application for classifying SKU product lines using a combination of business rules, exact/fuzzy matching, and machine learning. It is designed for Thermo Fisher Scientific's internal use to assist with SKU classification, leveraging both data-driven and rule-based approaches.

## Features

- **Exact Match:** Finds SKUs and names using robust substring matching.
- **Fuzzy Search:** Uses TF-IDF and cosine similarity for intelligent fuzzy matching.
- **Business Rules:** Extracts and applies business rules from training data and a rule book.
- **ML Fallback:** Trains a RandomForest model for fallback predictions.
- **Interactive UI:** Streamlit interface for easy input, results, and business rule explanations.
- **Downloadable Resources:** Allows downloading of the business rule book.

## How It Works

1. **Exact Match:** Attempts to find direct substring matches for SKU number and name.
2. **Fuzzy Search:** Uses TF-IDF vectorization and cosine similarity to find similar SKUs.
3. **Business Rule Extraction:** Analyzes SKU and name patterns to generate rules for each product line.
4. **ML Model:** Trains a RandomForest classifier for fallback predictions.
5. **Resource Download:** Provides a downloadable Excel file of business rules.

## Setup Instructions

### 1. Clone the Repository

```shell
git clone <your-repo-url>
cd Rule Engine
```

### 2. Install Requirements

Create a virtual environment (optional but recommended):

```shell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```shell
pip install -r requirements.txt
```

### 3. Prepare Data Files

- Place your training set Excel file at:
  ```
  Documents\Business Rule\Training_Set.xlsx
  ```
- Place your business rule book Excel file at:
  ```
  Documents\Business Rule\Business_Rule.xlsx
  ```
- Place your logo image as `logo2.png` in the project root.

### 4. Run the Application

```shell
streamlit run classification_bot.py
```

## Usage

- Enter a SKU number and/or SKU name in the input fields.
- Click "Classify SKU" to view predictions and business rule explanations.
- Download the business rule book from the sidebar.

## File Structure

- `classification_bot.py` — Main Streamlit application and logic.
- `requirements.txt` — Python dependencies.
- `Business_Rule.xlsx` — Business rule book (Excel).
- `Training_Set.xlsx` — Training data (Excel).
- `logo2.png` — Logo image.

## Requirements

See `requirements.txt` below.

## License

Internal use only. © Thermo Fisher Scientific.

---

## requirements.txt

````txt
streamlit
pandas
numpy
scikit-learn
fuzzywuzzy
python-Levenshtein
openpyxl