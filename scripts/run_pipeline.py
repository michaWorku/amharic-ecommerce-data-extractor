import argparse
import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1] # Adjust if your structure differs
sys.path.insert(0, str(project_root))

# Import modules from src
try:
    from src.data_ingestion.telegram_scraper import TelegramScraper
    from src.data_preprocessing.text_preprocessor import preprocess_dataframe
    from src.models.ner_trainer import train_ner_model 
    from src.models.model_evaluator import NERModelEvaluator 
    from src.analytics.vendor_scorecard import VendorScorecardGenerator 
    import pandas as pd
except ImportError as e:
    logger.error(f"Failed to import a necessary module. Ensure all dependencies are installed and "
                 f"the project structure is correct. Error: {e}")
    sys.exit(1)

# Define paths
RAW_DATA_PATH = project_root / 'data' / 'raw' / 'telegram_data.csv'
PREPROCESSED_DATA_PATH = project_root / 'data' / 'processed' / 'preprocessed_telegram_data.csv'
LABELED_DATA_PATH = project_root / 'data' / 'labeled' / '01_labeled_telegram_product_price_location.txt'
FINE_TUNED_MODEL_DIR = project_root / 'models' / 'fine_tuned_ner_model'
PREDICTED_LABELS_PATH = project_root / 'data' / 'predicted' / 'telegram_predictions.csv'
VENDOR_SCORECARD_PATH = project_root / 'data' / 'analytics' / 'vendor_scorecard.csv'
CHANNELS_FILE = project_root / 'config' / 'channels_to_crawl.txt'

# Ensure directories exist
for path in [RAW_DATA_PATH.parent, PREPROCESSED_DATA_PATH.parent, LABELED_DATA_PATH.parent,
             FINE_TUNED_MODEL_DIR, PREDICTED_LABELS_PATH.parent, VENDOR_SCORECARD_PATH.parent,
             CHANNELS_FILE.parent]:
    os.makedirs(path, exist_ok=True)

# Create a dummy channels file if it doesn't exist for initial setup ease
if not CHANNELS_FILE.exists():
    with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
        f.write("EthioMarketPlace\n")
        f.write("shega_shop_1\n")
        f.write("Ethio_online_store\n")
    logger.info(f"Created a dummy {CHANNELS_FILE} with example channels.")

def ingest_data_stage(limit: int):
    """
    Ingests data from Telegram channels using the TelegramScraper.
    """
    logger.info(f"--- Starting Data Ingestion Stage (Limit: {limit}) ---")
    scraper = TelegramScraper(
        api_id=os.getenv('TELEGRAM_API_ID'),
        api_hash=os.getenv('TELEGRAM_API_HASH'),
        channels_file_path=str(CHANNELS_FILE)
    )
    scraper.scrape_channels(limit=limit)
    logger.info("Data Ingestion Stage Complete.")

def preprocess_data_stage():
    """
    Preprocesses raw Telegram data.
    """
    logger.info("--- Starting Data Preprocessing Stage ---")
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data CSV not found at {RAW_DATA_PATH}. Please run 'ingest_data' stage first.")
        return

    try:
        df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        # Ensure 'message_text' column is handled as string for preprocessing
        df_raw['message_text'] = df_raw['message_text'].astype(str).fillna('')
        df_processed = preprocess_dataframe(df_raw, text_column='message_text', output_column='preprocessed_text')
        # Ensure 'views' column is numeric and fill NaN with 0 for consistency
        df_processed['views'] = pd.to_numeric(df_processed['views'], errors='coerce').fillna(0).astype(int)
        df_processed['message_date'] = pd.to_datetime(df_processed['message_date'], errors='coerce')
        df_processed.to_csv(PREPROCESSED_DATA_PATH, index=False)
        logger.info(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return
    logger.info("Data Preprocessing Stage Complete.")

def fine_tune_ner_stage(model_name: str, epochs: int, batch_size: int):
    """
    Fine-tunes the NER model.
    """
    logger.info(f"--- Starting NER Model Fine-tuning Stage (Model: {model_name}) ---")
    if not LABELED_DATA_PATH.exists():
        logger.error(f"Labeled data not found at {LABELED_DATA_PATH}. Please ensure you have a labeled dataset for fine-tuning.")
        logger.info("For demonstration, you can run `python src/models/ner_trainer.py` to create dummy labeled data and a test model.")
        return

    train_ner_model(
        labeled_data_path=str(LABELED_DATA_PATH),
        output_model_dir=str(FINE_TUNED_MODEL_DIR),
        model_name=model_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    logger.info("NER Model Fine-tuning Stage Complete.")

def evaluate_ner_stage():
    """
    Evaluates the fine-tuned NER model and performs interpretability analysis.
    """
    logger.info("--- Starting NER Model Evaluation Stage ---")
    if not FINE_TUNED_MODEL_DIR.exists() or not any(FINE_TUNED_MODEL_DIR.iterdir()):
        logger.error(f"Fine-tuned model not found at {FINE_TUNED_MODEL_DIR}. Please run 'fine_tune_ner' stage first.")
        return

    evaluator = NERModelEvaluator(model_dir=str(FINE_TUNED_MODEL_DIR))

    # Perform inference on preprocessed data and save
    if PREPROCESSED_DATA_PATH.exists():
        evaluator.predict_and_save_entities(
            preprocessed_data_path=str(PREPROCESSED_DATA_PATH),
            output_predictions_csv_path=str(PREDICTED_LABELS_PATH)
        )
    else:
        logger.warning(f"Preprocessed data not found at {PREPROCESSED_DATA_PATH}. Skipping entity prediction on full dataset.")

    # Perform interpretability analysis (requires a labeled evaluation dataset)
    # Re-use read_conll from conll_parser
    try:
        from src.utils.conll_parser import read_conll as read_conll_for_dataset
        from datasets import Dataset
        raw_data_for_eval_dataset = read_conll_for_dataset(str(LABELED_DATA_PATH))
        if raw_data_for_eval_dataset:
            # Need to derive id_to_label and label_to_id from the full labeled dataset
            all_labels_for_eval = sorted(list(set(item['label'] for sentence in raw_data_for_eval_dataset for item in sentence)))
            label_to_id_for_eval = {label: i for i, label in enumerate(all_labels_for_eval)}
            id_to_label_for_eval = {i: label for label, i in label_to_id_for_eval.items()}

            valid_tokens_for_eval = []
            valid_ner_tags_for_eval = []
            for sentence_data in raw_data_for_eval_dataset:
                tokens_sentence = [item['text'] for item in sentence_data]
                ner_tags_sentence = [label_to_id_for_eval[item['label']] for item in sentence_data]
                if tokens_sentence and ner_tags_sentence:
                    valid_tokens_for_eval.append(tokens_sentence)
                    valid_ner_tags_for_eval.append(ner_tags_sentence)

            eval_dataset_for_interp = Dataset.from_dict({"tokens": valid_tokens_for_eval, "ner_tags": valid_ner_tags_for_eval})

            # Pass the correct id_to_label and label_to_id to the evaluator
            evaluator.id_to_label = id_to_label_for_eval
            evaluator.label_to_id = label_to_id_for_eval

            evaluator.analyze_interpretability(eval_dataset=eval_dataset_for_interp, num_examples=5)
        else:
            logger.warning(f"No labeled data found at {LABELED_DATA_PATH} for interpretability analysis. Skipping.")
    except Exception as e:
        logger.error(f"Error preparing data for interpretability analysis: {e}")
        import traceback
        traceback.print_exc()

    logger.info("NER Model Evaluation Stage Complete.")

def generate_scorecards_stage():
    """
    Generates vendor scorecards using predicted entities and preprocessed data.
    """
    logger.info("--- Starting Vendor Scorecard Generation Stage ---")
    if not PREDICTED_LABELS_PATH.exists():
        logger.error(f"Predicted labels CSV not found at {PREDICTED_LABELS_PATH}. Please run 'evaluate_ner' stage first.")
        return
    if not PREPROCESSED_DATA_PATH.exists():
        logger.error(f"Preprocessed data CSV not found at {PREPROCESSED_DATA_PATH}. This is needed for original metadata.")
        return

    try:
        df_predictions = pd.read_csv(PREDICTED_LABELS_PATH, low_memory=False)
        df_preprocessed = pd.read_csv(PREPROCESSED_DATA_PATH, low_memory=False)

        # Ensure correct types and handle potential NaNs from CSV read
        df_predictions['token'] = df_predictions['token'].astype(str).fillna('')
        df_predictions['predicted_label'] = df_predictions['predicted_label'].astype(str).fillna('O')
        df_preprocessed['message_date'] = pd.to_datetime(df_preprocessed['message_date'], errors='coerce')
        df_preprocessed['views'] = pd.to_numeric(df_preprocessed['views'], errors='coerce').fillna(0).astype(int)
        df_preprocessed['channel_username'] = df_preprocessed['channel_username'].astype(str).fillna('unknown_channel')

        scorecard_generator = VendorScorecardGenerator()
        vendor_scorecard_df = scorecard_generator.generate_scorecard(df_predictions, df_preprocessed)
        vendor_scorecard_df.to_csv(VENDOR_SCORECARD_PATH, index=False)
        logger.info(f"Vendor scorecard saved to {VENDOR_SCORECARD_PATH}")
    except Exception as e:
        logger.error(f"Error during vendor scorecard generation: {e}")
        import traceback
        traceback.print_exc()
        return
    logger.info("Vendor Scorecard Generation Stage Complete.")

def main():
    parser = argparse.ArgumentParser(description="Run the Amharic E-commerce Data Extractor pipeline stages.")
    parser.add_argument('--stage', type=str, required=True,
                        choices=['ingest_data', 'preprocess_data', 'fine_tune_ner', 'evaluate_ner', 'generate_scorecards', 'all'],
                        help='Specify the pipeline stage to run.')
    parser.add_argument('--limit', type=int, default=500,
                        help='Limit the number of messages to scrape during data ingestion.')
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base",
                        help='Hugging Face model name for NER fine-tuning (e.g., xlm-roberta-base, bert-base-multilingual-cased).')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs for NER fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for NER fine-tuning.')

    args = parser.parse_args()

    if args.stage == 'ingest_data':
        ingest_data_stage(args.limit)
    elif args.stage == 'preprocess_data':
        preprocess_data_stage()
    elif args.stage == 'fine_tune_ner':
        fine_tune_ner_stage(args.model_name, args.epochs, args.batch_size)
    elif args.stage == 'evaluate_ner':
        evaluate_ner_stage()
    elif args.stage == 'generate_scorecards':
        generate_scorecards_stage()
    elif args.stage == 'all':
        ingest_data_stage(args.limit)
        preprocess_data_stage()
        # Note: 'all' stage for fine_tune_ner and evaluate_ner might be very time/resource intensive
        # For a full 'all' run, consider running fine_tune_ner and evaluate_ner separately first,
        # and only include if resources permit and you are sure labeled data is ready.
        # For now, I'll include them sequentially, assuming necessary data (e.g., labeled data) is ready.
        fine_tune_ner_stage(args.model_name, args.epochs, args.batch_size)
        evaluate_ner_stage()
        generate_scorecards_stage()
    else:
        logger.error(f"Unknown stage: {args.stage}")

if __name__ == '__main__':
    main()
