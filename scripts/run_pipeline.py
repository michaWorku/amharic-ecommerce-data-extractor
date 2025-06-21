import argparse
import asyncio
import pandas as pd
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules from src
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import the main function from the scraper, now capable of receiving a limit
from src.data_ingestion.telegram_scraper import main as run_scraper_main
from src.data_preprocessing.text_preprocessor import preprocess_dataframe


# Define paths relative to the project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, 'telegram_data.csv')
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_telegram_data.csv')


async def ingest_data_stage(message_limit: int = None):
    """
    Executes the data ingestion (scraping) process.
    Args:
        message_limit (int, optional): The maximum number of messages to scrape per channel.
    """
    logger.info(f"Starting data ingestion stage with message limit: {message_limit if message_limit is not None else 'None (all messages)'} per channel...")
    await run_scraper_main(message_limit=message_limit) # Pass the limit to the scraper's main function
    logger.info("Data ingestion stage completed.")

def preprocess_data_stage():
    """Executes the data preprocessing stage."""
    logger.info("Starting data preprocessing stage...")
    if not os.path.exists(RAW_CSV_PATH):
        logger.error(f"Raw data CSV not found at '{RAW_CSV_PATH}'. Please run 'ingest_data' first.")
        return

    try:
        df = pd.read_csv(RAW_CSV_PATH, encoding='utf-8')
        logger.info(f"Loaded {len(df)} rows from raw data.")
        
        processed_df = preprocess_dataframe(df.copy())

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        processed_df.to_csv(PROCESSED_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Processed data saved to '{PROCESSED_CSV_PATH}'.")

    except FileNotFoundError:
        logger.error(f"Error: Raw data file not found at {RAW_CSV_PATH}")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")

# Placeholder for future stages
def fine_tune_ner_stage():
    """Placeholder for the NER model fine-tuning stage."""
    logger.info("Starting NER model fine-tuning stage (Not yet implemented).")
    pass

def generate_scorecards_stage():
    """Placeholder for the vendor scorecard generation stage."""
    logger.info("Starting vendor scorecard generation stage (Not yet implemented).")
    pass


async def main():
    parser = argparse.ArgumentParser(description="Run various stages of the Amharic E-commerce Data Extractor pipeline.")
    parser.add_argument('--stage', type=str, required=True,
                        choices=['ingest_data', 'preprocess_data', 'fine_tune_ner', 'generate_scorecards'],
                        help="Specify the pipeline stage to run.")
    
    # Add optional argument for message limit, specifically for ingest_data stage
    parser.add_argument('--limit', type=int, default=None,
                        help="Maximum number of messages to scrape per channel (only for ingest_data stage).")

    args = parser.parse_args()

    if args.stage == 'ingest_data':
        # Pass the limit argument to the ingest_data_stage function
        await ingest_data_stage(message_limit=args.limit)
    elif args.stage == 'preprocess_data':
        preprocess_data_stage()
    elif args.stage == 'fine_tune_ner':
        fine_tune_ner_stage()
    elif args.stage == 'generate_scorecards':
        generate_scorecards_stage()
    else:
        logger.error("Invalid stage specified.")

if __name__ == '__main__':
    asyncio.run(main())

