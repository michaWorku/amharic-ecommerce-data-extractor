import asyncio
import csv
import os
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument, Channel
from dotenv import load_dotenv
import logging
import pandas as pd # Import pandas for data summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv('.env')

# Telegram API credentials
# Ensure these are set in your .env file
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
PHONE = os.getenv('PHONE_NUMBER') # Optional: Only needed if you plan to use a user account, not just public channels.

# Get the directory of the current script (src/data_ingestion/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to reach the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(current_script_dir))

# Define paths relative to the project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MEDIA_DIR = os.path.join(RAW_DATA_DIR, 'media')
CONFIG_DIR = os.path.join(DATA_DIR, 'config')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
# os.makedirs(MEDIA_DIR, exist_ok=True) # Media download is skipped, so no need to ensure this dir exists for now

# Path for the CSV file where scraped data will be stored
OUTPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, 'telegram_data.csv')
CHANNELS_LIST_PATH = os.path.join(CONFIG_DIR, 'channels_to_crawl.txt')

# Initialize the Telegram client
client = TelegramClient('scraping_session', API_ID, API_HASH)

async def scrape_channel(client_instance: TelegramClient, channel_username: str, writer, media_base_dir: str, message_limit: int = None):
    """
    Scrapes messages from a single Telegram channel and writes data to a CSV.
    **Does NOT download media files** due to time constraints, but notes their presence.

    Args:
        client_instance (TelegramClient): The initialized Telethon client.
        channel_username (str): The username of the Telegram channel (e.g., '@Shageronlinestore').
        writer (csv.writer): The CSV writer object to write rows.
        media_base_dir (str): The base directory to save downloaded media files (no longer used for download).
        message_limit (int, optional): The maximum number of messages to scrape per channel.
                                       If None, scrapes all available messages (up to Telethon's default).
    """
    try:
        entity = await client_instance.get_entity(channel_username)

        if isinstance(entity, Channel):
            channel_title = entity.title if entity.title is not None else '' # Ensure channel_title is string
            logger.info(f"Scraping data from channel: '{channel_title}' (@{channel_username})")
        else:
            logger.warning(f"'{channel_username}' is not a public channel or supergroup. Skipping.")
            return

        # Use the provided message_limit
        async for message in client_instance.iter_messages(entity, limit=message_limit):
            # Convert message text to empty string if None
            message_text = message.message if message.message is not None else ''
            # Convert date to ISO format string, or empty string if None
            message_date = message.date.isoformat() if message.date else ''
            message_id = message.id # message.id should always be present

            sender_id = message.sender_id
            # Ensure sender_username is an empty string if not found
            sender_username = getattr(message.sender, 'username', '')

            media_path = '' # Default to empty string
            media_type = '' # Default to empty string

            if message.media:
                # Media is present, but we are explicitly skipping download.
                # Still record the type if it's a photo or document.
                if isinstance(message.media, MessageMediaPhoto):
                    media_type = 'photo_skipped_download' # Indicate media type but note no download
                    media_path = '' # No path as not downloaded
                    # logger.info(f"Media (photo) found for message {message.id} in {channel_username}, skipping download.") # Reduced logging volume
                elif isinstance(message.media, MessageMediaDocument):
                    media_type = 'document_skipped_download' # Indicate media type but note no download
                    media_path = '' # No path as not downloaded
                    # logger.info(f"Media (document) found for message {message.id} in {channel_username}, skipping download.") # Reduced logging volume
                else:
                    # For other unknown media types, just log and keep path/type empty
                    media_type = f"unsupported_media_type_{type(message.media).__name__}"
                    media_path = ''
                    logger.warning(f"Unsupported media type '{type(message.media).__name__}' for message {message.id} in {channel_username}, skipping download.")


            # Get message views if available, default to 0 if None
            views = getattr(message, 'views', 0)

            # Write the collected data to the CSV file
            writer.writerow([
                channel_title,
                channel_username,
                message_id,
                message_text,
                message_date,
                sender_id,
                sender_username,
                media_path, # This will now consistently be '' if no download
                media_type, # This will indicate type_skipped_download or ''
                views
            ])
            
    except ValueError as ve:
        logger.error(f"Could not find entity for channel '{channel_username}': {ve}")
    except SessionPasswordNeededError:
        logger.error("Session password needed. Please ensure you are logged in or using a bot token.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while scraping '{channel_username}': {e}")


async def main(message_limit: int = None):
    """
    Main function to orchestrate the scraping process.
    Reads channels from a CSV, scrapes data, and stores it.
    Includes summary statistics generation at the end.

    Args:
        message_limit (int, optional): The maximum number of messages to scrape per channel.
                                       If None, scrapes all available messages.
    """
    if not all([API_ID, API_HASH]):
        logger.error("TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env file. Please set them.")
        return

    await client.start()

    # Check if the file exists and is not empty to determine if header needs to be written
    file_exists = os.path.exists(OUTPUT_CSV_PATH)
    file_is_empty = not file_exists or os.stat(OUTPUT_CSV_PATH).st_size == 0

    # Open the CSV file in append mode ('a')
    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row ONLY if the file was empty or didn't exist
        if file_is_empty:
            writer.writerow([
                'channel_title', 'channel_username', 'message_id', 'message_text',
                'message_date', 'sender_id', 'sender_username', 'media_path', 'media_type', 'views'
            ])
            logger.info("CSV header written.")
        else:
            logger.info("CSV file already exists and is not empty. Appending data without writing header.")


        channels_to_scrape = []
        try:
            with open(CHANNELS_LIST_PATH, 'r', encoding='utf-8') as channels_file:
                reader = csv.reader(channels_file)
                for row in reader:
                    if row and row[0].strip(): # Ensure row is not empty AND channel username is not empty
                        channels_to_scrape.append(row[0].strip())
            if not channels_to_scrape:
                logger.warning(f"No valid channels found in '{CHANNELS_LIST_PATH}'. Please add channel usernames.")
                return # This return is a potential early exit if the CSV is genuinely empty or malformed
        except FileNotFoundError:
            logger.error(f"Channels list file not found: '{CHANNELS_LIST_PATH}'. Please create it.")
            return
        except Exception as e:
            logger.error(f"Error reading channels from CSV: {e}")
            return

        logger.info(f"Attempting to scrape {len(channels_to_scrape)} channels: {channels_to_scrape}") # Added logging
        for channel in channels_to_scrape:
            await scrape_channel(client, channel, writer, MEDIA_DIR, message_limit=message_limit)
            logger.info(f"Finished scraping data from {channel}")

    logger.info(f"Scraping complete. Data saved to '{OUTPUT_CSV_PATH}'.")

    await client.disconnect()

    # --- Generate and Print Summary Statistics ---
    logger.info("Generating summary statistics for scraped data...")
    if os.path.exists(OUTPUT_CSV_PATH) and os.stat(OUTPUT_CSV_PATH).st_size > 0:
        try:
            df_scraped = pd.read_csv(OUTPUT_CSV_PATH, encoding='utf-8')

            # --- Overall Summary ---
            num_channels_scraped = df_scraped['channel_username'].nunique()
            total_messages_scraped = len(df_scraped)

            print("\n--- Overall Scraped Data Summary ---")
            print(f"Total Channels Scraped: {num_channels_scraped}")
            print(f"Total Messages Scraped: {total_messages_scraped}")
            print("\n--- Missing Values Count (Overall) ---")
            print(df_scraped.isnull().sum()[df_scraped.isnull().sum() > 0].to_string()) # Only show columns with missing values

            # --- Per-Channel Summary ---
            print("\n--- Per-Channel Scraped Data Summary ---")
            
            # Messages per channel
            messages_per_channel = df_scraped['channel_username'].value_counts().rename('Total Messages')
            print("\nMessages per Channel:")
            print(messages_per_channel.to_string())

            # Missing values per channel for relevant columns
            # Ensure the columns exist before attempting to group/sum
            columns_to_check_for_nan = [
                'message_text', 'message_date', 'views', 'media_path', 'media_type', 'sender_username'
            ]
            
            # Filter columns_to_check_for_nan to only include those actually in df_scraped
            existing_cols_to_check = [col for col in columns_to_check_for_nan if col in df_scraped.columns]

            if existing_cols_to_check:
                missing_values_per_channel = df_scraped.groupby('channel_username')[existing_cols_to_check].apply(lambda x: x.isnull().sum())
                print("\nMissing Values Per Channel:")
                print(missing_values_per_channel.to_string())
            else:
                print("\nNo relevant columns found for missing value check per channel.")

            # Optional: Average views per channel
            if 'views' in df_scraped.columns:
                avg_views_per_channel = df_scraped.groupby('channel_username')['views'].mean().rename('Average Views')
                print("\nAverage Views Per Channel:")
                print(avg_views_per_channel.to_string(float_format="%.2f"))

        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
    else:
        logger.info("No data scraped or CSV file is empty. Skipping summary statistics.")


if __name__ == '__main__':
    asyncio.run(main())
