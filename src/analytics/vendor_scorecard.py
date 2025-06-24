import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict

# Import Hugging Face components for loading the fine-tuned model
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger(__name__)

def extract_entities_by_type(entities: List[Dict], entity_type: str) -> List[str]:
    """Extracts all entities of a specific type from a list of NER results."""
    return [ent['word'] for ent in entities if ent['entity_group'] == entity_type]

def extract_numerical_price(price_tokens: List[str]) -> float:
    """
    Extracts and converts a price entity (e.g., ['5000', 'ብር']) into a float.
    Handles commas, currency symbols, and ensures valid conversion.
    """
    if not price_tokens:
        return np.nan
    
    full_price_str = "".join(price_tokens).lower()
    price_value_str = re.sub(r'[ብርbirr\s,]', '', full_price_str)
    
    try:
        return float(price_value_str)
    except ValueError:
        logger.warning(f"Could not convert price '{full_price_str}' to float. Returning NaN.")
        return np.nan

def calculate_vendor_metrics(df_vendor: pd.DataFrame, ner_pipeline) -> Dict:
    """
    Calculates key performance metrics for a single vendor, including NER inference.

    Args:
        df_vendor (pd.DataFrame): DataFrame containing posts for a single vendor.
        ner_pipeline: The Hugging Face NER pipeline for entity extraction.

    Returns:
        Dict: A dictionary of calculated metrics for the vendor.
    """
    if df_vendor.empty:
        return {
            'Avg. Views/Post': 0,
            'Posts/Week': 0,
            'Avg. Price (ETB)': 0,
            'Top Product': 'N/A',
            'Top Product Price': 'N/A',
            'Total Posts': 0,
            'Date Range Days': 0
        }

    # Perform NER inference on this vendor's messages
    ner_results = []
    for text in df_vendor['preprocessed_text']:
        if pd.isna(text) or not text.strip():
            ner_results.append([])
            continue
        try:
            results = ner_pipeline(text)
            ner_results.append(results)
        except Exception as e:
            logger.warning(f"Error during NER for vendor message: {e}. Appending empty result.")
            ner_results.append([])
    df_vendor['extracted_entities'] = ner_results

    # Extract specific entity types
    df_vendor['products'] = df_vendor['extracted_entities'].apply(lambda x: extract_entities_by_type(x, 'PRODUCT'))
    df_vendor['prices'] = df_vendor['extracted_entities'].apply(lambda x: extract_entities_by_type(x, 'PRICE'))
    df_vendor['all_numerical_prices'] = df_vendor['prices'].apply(lambda x: [extract_numerical_price([p]) for p in x if p])
    df_vendor['all_numerical_prices'] = df_vendor['all_numerical_prices'].apply(lambda x: [val for val in x if not pd.isna(val)])

    # Total Posts
    total_posts = len(df_vendor)

    # Average Views per Post
    avg_views_per_post = df_vendor['views'].mean() if not df_vendor['views'].empty else 0

    # Posting Frequency (Posts per Week)
    min_date = df_vendor['date'].min()
    max_date = df_vendor['date'].max()
    date_range_days = (max_date - min_date).days + 1
    
    if date_range_days <= 0:
        posting_frequency = total_posts
        date_range_days = 1
    else:
        posting_frequency = total_posts / (date_range_days / 7)

    # Average Price Point
    all_prices_flat = [price for sublist in df_vendor['all_numerical_prices'] for price in sublist]
    avg_price_point = np.mean(all_prices_flat) if all_prices_flat else np.nan

    # Top Performing Post
    top_post = df_vendor.loc[df_vendor['views'].idxmax()]
    top_product = top_post['products'][0] if top_post['products'] else 'N/A'
    top_product_price = top_post['all_numerical_prices'][0] if top_post['all_numerical_prices'] else np.nan
    
    return {
        'Total Posts': total_posts,
        'Avg. Views/Post': avg_views_per_post,
        'Posts/Week': posting_frequency,
        'Avg. Price (ETB)': avg_price_point,
        'Top Product': top_product,
        'Top Product Price': top_product_price,
        'Date Range Days': date_range_days
    }


def generate_vendor_scorecard(preprocessed_data_path: Path, model_path_or_name: str, output_scorecard_path: Path):
    """
    Generates the FinTech Vendor Scorecard by loading preprocessed data,
    running NER inference, calculating vendor metrics, and a lending score.

    Args:
        preprocessed_data_path (Path): Path to the preprocessed CSV data.
        model_path_or_name (str): Path to the fine-tuned NER model (local or HF Hub name).
        output_scorecard_path (Path): Path to save the final vendor scorecard CSV.
    """
    logger.info(f"Generating vendor scorecard using data from: {preprocessed_data_path}")
    logger.info(f"Loading NER model from: {model_path_or_name}")

    if not preprocessed_data_path.exists():
        logger.error(f"Error: Preprocessed data not found at {preprocessed_data_path}.")
        raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_data_path}")

    try:
        df = pd.read_csv(preprocessed_data_path)
        logger.info(f"Loaded {len(df)} rows from {preprocessed_data_path}")
        required_cols = ['preprocessed_text', 'views', 'date', 'channel']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing one or more required columns ({required_cols}) in the dataframe.")
            raise ValueError("Essential metadata columns are missing in the input DataFrame.")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['views'] = pd.to_numeric(df['views'], errors='coerce')
        df.dropna(subset=['date', 'views', 'channel', 'preprocessed_text'], inplace=True)
        logger.info(f"DataFrame after dropping rows with missing essential metadata: {len(df)} rows.")

    except Exception as e:
        logger.error(f"Failed to load or validate preprocessed data: {e}")
        raise

    # Load NER model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        model = AutoModelForTokenClassification.from_pretrained(model_path_or_name)
        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        logger.info("Fine-tuned NER model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model or create pipeline from '{model_path_or_name}': {e}")
        logger.error("Please verify the model path/name and ensure it exists locally or on Hugging Face Hub.")
        raise

    logger.info("Calculating metrics for all unique vendors...")
    vendor_metrics_list = []
    unique_vendors = df['channel'].unique()

    for vendor_name in unique_vendors:
        df_vendor = df[df['channel'] == vendor_name].copy()
        metrics = calculate_vendor_metrics(df_vendor, ner_pipeline) # Pass ner_pipeline here
        metrics['Vendor'] = vendor_name
        vendor_metrics_list.append(metrics)

    vendor_scorecard_df = pd.DataFrame(vendor_metrics_list)
    logger.info("Vendor metrics calculated.")

    # Normalize metrics and calculate Lending Score
    METRIC_WEIGHTS = {
        'Avg. Views/Post': 0.4,
        'Posts/Week': 0.4,
        'Avg. Price (ETB)': 0.2,
    }

    for col in METRIC_WEIGHTS.keys():
        if col not in vendor_scorecard_df.columns:
            vendor_scorecard_df[col] = 0.0
        vendor_scorecard_df[col] = vendor_scorecard_df[col].fillna(0)

    normalized_df = vendor_scorecard_df.copy()
    for metric, weight in METRIC_WEIGHTS.items():
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()
        
        if max_val == min_val:
            normalized_df[f'Normalized {metric}'] = 0.0
        else:
            normalized_df[f'Normalized {metric}'] = (normalized_df[metric] - min_val) / (max_val - min_val)
            
    normalized_df['Lending Score'] = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        normalized_df['Lending Score'] += normalized_df[f'Normalized {metric}'] * weight

    max_possible_score = sum(METRIC_WEIGHTS.values())
    normalized_df['Lending Score (0-100)'] = (normalized_df['Lending Score'] / max_possible_score) * 100
    logger.info("Lending Score calculated.")

    final_scorecard_display = normalized_df[[
        'Vendor',
        'Avg. Views/Post',
        'Posts/Week',
        'Avg. Price (ETB)',
        'Top Product',
        'Top Product Price',
        'Lending Score (0-100)'
    ]].sort_values(by='Lending Score (0-100)', ascending=False)
    
    os.makedirs(output_scorecard_path.parent, exist_ok=True)
    final_scorecard_display.to_csv(output_scorecard_path, index=False, encoding='utf-8')
    logger.info(f"Vendor scorecard saved to: {output_scorecard_path}")

# Example usage (for testing this module directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define paths relative to this script for testing
    current_script_dir = Path(__file__).parent
    project_root = current_script_dir.parent.parent # Adjust if different structure
    
    test_preprocessed_data_path = project_root / 'data' / 'processed' / 'preprocessed_telegram_data.csv'
    # Use the local model directory for testing if it exists, otherwise a placeholder HF name
    test_model_path_or_name = project_root / 'fine_tuned_ner_model'
    if not test_model_path_or_name.exists():
        test_model_path_or_name = "xlm-roberta-base" # Fallback for testing if local model not saved
        logger.warning(f"Local fine-tuned model not found at {test_model_path_or_name}. Using '{test_model_path_or_name}' from Hugging Face Hub for testing. This might download the base model if not cached.")

    test_output_scorecard_path = project_root / 'data' / 'processed' / 'vendor_scorecard_test.csv'

    try:
        generate_vendor_scorecard(
            preprocessed_data_path=test_preprocessed_data_path,
            model_path_or_name=str(test_model_path_or_name),
            output_scorecard_path=test_output_scorecard_path
        )
        logger.info("Vendor scorecard generation test completed successfully.")
    except Exception as e:
        logger.error(f"Vendor scorecard generation test failed: {e}")

