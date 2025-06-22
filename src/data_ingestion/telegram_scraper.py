import asyncio
import csv
import os
from datetime import datetime
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument, Channel
from dotenv import load_dotenv
import logging


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

# Correct BASE_DIR calculation for nested script
# Get the directory of the current script (src/data_ingestion/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(current_script_dir))

# Define paths relative to the project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MEDIA_DIR = os.path.join(RAW_DATA_DIR, 'media')
CONFIG_DIR = os.path.join(DATA_DIR, 'config')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

# Path for the CSV file where scraped data will be stored
OUTPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, 'telegram_data.csv')
CHANNELS_LIST_PATH = os.path.join(CONFIG_DIR, 'channels_to_crawl.csv')

# Initialize the Telegram client (no need to redefine if already global for the module)
client = TelegramClient('scraping_session', API_ID, API_HASH)

async def scrape_channel(client_instance: TelegramClient, channel_username: str, writer, media_base_dir: str, message_limit: int = None):
    """
    Scrapes messages from a single Telegram channel and writes data to a CSV.
    Downloads media (photos/documents) to a specified directory.

    Args:
        client_instance (TelegramClient): The initialized Telethon client.
        channel_username (str): The username of the Telegram channel (e.g., '@Shageronlinestore').
        writer (csv.writer): The CSV writer object to write rows.
        media_base_dir (str): The base directory to save downloaded media files.
        message_limit (int, optional): The maximum number of messages to scrape per channel.
                                       If None, scrapes all available messages (up to Telethon's default).
    """
    try:
        entity = await client_instance.get_entity(channel_username)
        
        if isinstance(entity, Channel):
            channel_title = entity.title
            logger.info(f"Scraping data from channel: '{channel_title}' (@{channel_username})")
        else:
            logger.warning(f"'{channel_username}' is not a public channel or supergroup. Skipping.")
            return

        # Use the provided message_limit
        async for message in client_instance.iter_messages(entity, limit=message_limit):
            message_text = message.message
            message_date = message.date.isoformat() if message.date else None
            message_id = message.id
            
            sender_id = message.sender_id
            sender_username = ''
            if message.sender:
                sender_username = getattr(message.sender, 'username', '')

            media_path = None
            media_type = None

            if message.media:
                if isinstance(message.media, MessageMediaPhoto):
                    media_type = 'photo'
                    filename = f"{channel_username}_{message.id}_photo.jpg"
                    media_full_path = os.path.join(media_base_dir, filename)
                    try:
                        await client_instance.download_media(message.media, media_full_path)
                        media_path = os.path.relpath(media_full_path, RAW_DATA_DIR)
                        logger.info(f"Downloaded photo: {media_path}")
                    except Exception as e:
                        logger.error(f"Error downloading photo for message {message.id} in {channel_username}: {e}")
                        media_path = None

                elif isinstance(message.media, MessageMediaDocument):
                    media_type = 'document'
                    document_filename = 'document'
                    for attr in message.media.document.attributes:
                        if hasattr(attr, 'file_name'):
                            document_filename = attr.file_name
                            break
                    filename = f"{channel_username}_{message.id}_{document_filename}"
                    media_full_path = os.path.join(media_base_dir, filename)
                    try:
                        await client_instance.download_media(message.media, media_full_path)
                        media_path = os.path.relpath(media_full_path, RAW_DATA_DIR)
                        logger.info(f"Downloaded document: {media_path}")
                    except Exception as e:
                        logger.error(f"Error downloading document for message {message.id} in {channel_username}: {e}")
                        media_path = None

            views = getattr(message, 'views', None)

            writer.writerow([
                channel_title,
                channel_username,
                message_id,
                message_text,
                message_date,
                sender_id,
                sender_username,
                media_path,
                media_type,
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

    Args:
        message_limit (int, optional): The maximum number of messages to scrape per channel.
                                       If None, scrapes all available messages.
    """
    if not all([API_ID, API_HASH]):
        logger.error("TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env file. Please set them.")
        return

    await client.start()

    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'channel_title', 'channel_username', 'message_id', 'message_text',
            'message_date', 'sender_id', 'sender_username', 'media_path', 'media_type', 'views'
        ])

        channels_to_scrape = []
        try:
            with open(CHANNELS_LIST_PATH, 'r', encoding='utf-8') as channels_file:
                reader = csv.reader(channels_file)
                next(reader) # Skip header row
                for row in reader:
                    if row:
                        channels_to_scrape.append(row[0].strip())
            if not channels_to_scrape:
                logger.warning(f"No channels found in '{CHANNELS_LIST_PATH}'. Please add channel usernames.")
                return
        except FileNotFoundError:
            logger.error(f"Channels list file not found: '{CHANNELS_LIST_PATH}'. Please create it.")
            return
        except Exception as e:
            logger.error(f"Error reading channels from CSV: {e}")
            return

        for channel in channels_to_scrape:
            # Pass the message_limit to scrape_channel
            await scrape_channel(client, channel, writer, MEDIA_DIR, message_limit=message_limit)
            logger.info(f"Finished scraping data from {channel}")

    logger.info(f"Scraping complete. Data saved to '{OUTPUT_CSV_PATH}' and media to '{MEDIA_DIR}'.")

    await client.disconnect()

if __name__ == '__main__':
    # This block is mainly for direct testing of the scraper.
    # When run via run_pipeline.py, the limit will be passed by it.
    # For standalone testing, you can uncomment and set a limit here:
    # asyncio.run(main(message_limit=100)) # Scrape max 100 messages per channel for testing
    asyncio.run(main()) # No limit by default if run standalone
