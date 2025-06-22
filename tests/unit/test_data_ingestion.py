import pytest
import asyncio
import os
import csv
from unittest.mock import AsyncMock, MagicMock, patch, mock_open, call # Import 'call' for specific assertions
from datetime import datetime # Import datetime explicitly

# Adjust sys.path to allow imports from src
import sys
from pathlib import Path
# Assuming tests/unit is two levels down from project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_ingestion.telegram_scraper import scrape_channel, main as scraper_main, OUTPUT_CSV_PATH, CHANNELS_LIST_PATH, MEDIA_DIR, RAW_DATA_DIR
from telethon.tl.types import Message, User, Channel, MessageMediaPhoto, MessageMediaDocument, DocumentAttributeFilename

# Define common data for mocking
MOCKED_CHANNEL_TITLE = "Test Channel"
MOCKED_CHANNEL_USERNAME = "@testchannel"
MOCKED_MESSAGE_TEXT = "Test message content."
MOCKED_MESSAGE_DATE = datetime(2025, 6, 21, 10, 0, 0)
MOCKED_SENDER_ID = 12345
MOCKED_SENDER_USERNAME = "test_user"
MOCKED_VIEWS = 100

# Helper function to create mock Telethon Message objects
def create_mock_message(id=123, has_media=False, is_photo=False, is_document=False, has_views=True, message_text=MOCKED_MESSAGE_TEXT):
    """Creates a mock Telethon Message object."""
    mock_msg = AsyncMock(spec=Message)
    mock_msg.id = id # Allow varying message IDs for multiple messages in a test
    mock_msg.message = message_text
    mock_msg.date = MOCKED_MESSAGE_DATE
    
    mock_sender = MagicMock(spec=User)
    mock_sender.username = MOCKED_SENDER_USERNAME
    mock_msg.sender = mock_sender

    mock_msg.media = None
    if has_media:
        if is_photo:
            mock_msg.media = MagicMock(spec=MessageMediaPhoto)
            mock_msg.media.photo = MagicMock() # Simulate a photo object
        elif is_document:
            mock_msg.media = MagicMock(spec=MessageMediaDocument)
            # Ensure document has attributes for filename, even if mocked
            mock_msg.media.document = MagicMock()
            mock_msg.media.document.attributes = [MagicMock(spec=DocumentAttributeFilename, file_name='test_doc.pdf')]
    
    if has_views:
        mock_msg.views = MOCKED_VIEWS
    else:
        mock_msg.views = None # Explicitly set to None if no views

    return mock_msg

# Async Iterator Helper for mocking async for loops
class AsyncMockIterator:
    """A helper class to make an iterable behave like an async iterator for mocks."""
    def __init__(self, seq, limit=None):
        # Apply the limit directly to the sequence provided to the iterator
        self.seq = seq[:limit] if limit is not None else seq
        self.iter = iter(self.seq)

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration

    def __aiter__(self):
        return self

@pytest.fixture
def mock_telethon_client():
    """Mocks the global Telethon client instance."""
    with patch('src.data_ingestion.telegram_scraper.TelegramClient') as MockClient:
        mock_instance = MockClient.return_value
        # Ensure client.start() and client.disconnect() are awaitable
        mock_instance.start = AsyncMock()
        mock_instance.disconnect = AsyncMock()
        mock_instance.get_entity = AsyncMock()
        # By default, iter_messages will return an empty async iterator.
        # Specific tests will override side_effect to yield messages.
        mock_instance.iter_messages = MagicMock(side_effect=lambda *args, **kwargs: AsyncMockIterator([]))
        mock_instance.download_media = AsyncMock()
        yield mock_instance

@pytest.fixture
def mock_csv_writer():
    """Mocks the csv.writer to capture written rows."""
    with patch('src.data_ingestion.telegram_scraper.csv.writer') as MockWriter:
        mock_writer_instance = MockWriter.return_value
        yield mock_writer_instance

@pytest.fixture
def mock_os_functions():
    """Mocks os-related functions like os.makedirs and os.path.exists."""
    with patch('src.data_ingestion.telegram_scraper.os.makedirs') as mock_makedirs, \
         patch('src.data_ingestion.telegram_scraper.os.path.exists') as mock_exists, \
         patch('src.data_ingestion.telegram_scraper.os.stat') as mock_stat:
        
        mock_exists.return_value = True # Assume files exist by default, override as needed
        mock_makedirs.return_value = None # makedirs doesn't return anything
        mock_stat.return_value.st_size = 100 # Assume file is not empty by default for append logic
        yield {
            "makedirs": mock_makedirs,
            "exists": mock_exists,
            "stat": mock_stat
        }


@pytest.mark.asyncio
async def test_scrape_channel_text_only(mock_telethon_client, mock_csv_writer, mock_os_functions):
    """Tests scraping a channel with text-only messages."""
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title=MOCKED_CHANNEL_TITLE)
    
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator([
        create_mock_message(message_text="Hello world.", id=1),
        create_mock_message(message_text="Another message.", id=2),
    ], limit=limit) # Pass limit to AsyncMockIterator

    await scrape_channel(mock_telethon_client, MOCKED_CHANNEL_USERNAME, mock_csv_writer, MEDIA_DIR)

    mock_telethon_client.get_entity.assert_called_once_with(MOCKED_CHANNEL_USERNAME)
    assert mock_csv_writer.writerow.call_count == 2
    
    expected_row1 = [
        MOCKED_CHANNEL_TITLE, MOCKED_CHANNEL_USERNAME, 1, "Hello world.",
        MOCKED_MESSAGE_DATE.isoformat(), MOCKED_SENDER_ID, MOCKED_SENDER_USERNAME, None, None, MOCKED_VIEWS
    ]
    expected_row2 = [
        MOCKED_CHANNEL_TITLE, MOCKED_CHANNEL_USERNAME, 2, "Another message.",
        MOCKED_MESSAGE_DATE.isoformat(), MOCKED_SENDER_ID, MOCKED_SENDER_USERNAME, None, None, MOCKED_VIEWS
    ]

    # Get the actual calls made to writerow
    actual_calls_args = [call.args[0] for call in mock_csv_writer.writerow.call_args_list]
    
    # Assert that the expected rows are present in the actual calls
    assert expected_row1 in actual_calls_args
    assert expected_row2 in actual_calls_args

    mock_telethon_client.download_media.assert_not_called()


@pytest.mark.asyncio
async def test_scrape_channel_with_photo(mock_telethon_client, mock_csv_writer, mock_os_functions):
    """Tests scraping a channel with messages including photos."""
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title=MOCKED_CHANNEL_TITLE)
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator([
        create_mock_message(has_media=True, is_photo=True, message_text="Check out this product!"),
    ], limit=limit)
    
    await scrape_channel(mock_telethon_client, MOCKED_CHANNEL_USERNAME, mock_csv_writer, MEDIA_DIR)

    mock_telethon_client.download_media.assert_called_once()
    args, _ = mock_telethon_client.download_media.call_args
    download_path = args[1]
    assert f"{MOCKED_CHANNEL_USERNAME}_123_photo.jpg" in download_path
    assert MEDIA_DIR in download_path

    mock_csv_writer.writerow.assert_called_once()
    args, _ = mock_csv_writer.writerow.call_args
    written_row = args[0]
    assert written_row[7].startswith('media/')
    assert written_row[8] == 'photo'

@pytest.mark.asyncio
async def test_scrape_channel_with_document(mock_telethon_client, mock_csv_writer, mock_os_functions):
    """Tests scraping a channel with messages including documents."""
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title=MOCKED_CHANNEL_TITLE)
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator([
        create_mock_message(has_media=True, is_document=True, message_text="See our catalog."),
    ], limit=limit)
    
    await scrape_channel(mock_telethon_client, MOCKED_CHANNEL_USERNAME, mock_csv_writer, MEDIA_DIR)

    mock_telethon_client.download_media.assert_called_once()
    args, _ = mock_telethon_client.download_media.call_args
    download_path = args[1]
    assert f"{MOCKED_CHANNEL_USERNAME}_123_test_doc.pdf" in download_path
    assert MEDIA_DIR in download_path

    mock_csv_writer.writerow.assert_called_once()
    args, _ = mock_csv_writer.writerow.call_args
    written_row = args[0]
    assert written_row[7].startswith('media/')
    assert written_row[8] == 'document'

@pytest.mark.asyncio
async def test_scrape_channel_no_views(mock_telethon_client, mock_csv_writer, mock_os_functions):
    """Tests scraping a channel with messages that have no views information."""
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title=MOCKED_CHANNEL_TITLE)
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator([
        create_mock_message(has_views=False, message_text="A message without views."),
    ], limit=limit)

    await scrape_channel(mock_telethon_client, MOCKED_CHANNEL_USERNAME, mock_csv_writer, MEDIA_DIR)
    
    mock_csv_writer.writerow.assert_called_once()
    args, _ = mock_csv_writer.writerow.call_args
    written_row = args[0]
    assert written_row[9] is None # Views should be None

@pytest.mark.asyncio
async def test_scrape_channel_limit(mock_telethon_client, mock_csv_writer, mock_os_functions):
    """Tests if message_limit is passed correctly to iter_messages and respected."""
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title=MOCKED_CHANNEL_TITLE)
    
    messages_to_yield = [
        create_mock_message(id=1, message_text="msg1"),
        create_mock_message(id=2, message_text="msg2"),
        create_mock_message(id=3, message_text="msg3"),
    ]
    # This side_effect passes the limit received by iter_messages to AsyncMockIterator
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator(messages_to_yield, limit=limit)

    await scrape_channel(mock_telethon_client, MOCKED_CHANNEL_USERNAME, mock_csv_writer, MEDIA_DIR, message_limit=2)

    # iter_messages should be called once with the correct limit argument
    mock_telethon_client.iter_messages.assert_called_once_with(
        mock_telethon_client.get_entity.return_value, # The entity mock returned by get_entity
        limit=2
    )
    # The csv_writer should have written only 2 messages due to the limit being respected by AsyncMockIterator
    assert mock_csv_writer.writerow.call_count == 2
    # Optionally, verify the content of the two calls
    actual_calls_args = [call.args[0] for call in mock_csv_writer.writerow.call_args_list]
    assert create_mock_message(id=1, message_text="msg1").message == actual_calls_args[0][3]
    assert create_mock_message(id=2, message_text="msg2").message == actual_calls_args[1][3]


@pytest.mark.asyncio
@patch('src.data_ingestion.telegram_scraper.csv.writer') # Patch csv.writer at module level for main
@patch('src.data_ingestion.telegram_scraper.open', new_callable=mock_open) # Patch builtins.open
async def test_scraper_main_initial_run_writes_header(mock_open_func, mock_csv_writer_class, mock_telethon_client, mock_os_functions):
    """Tests main function when CSV file does not exist (initial run)."""
    mock_os_functions["exists"].side_effect = lambda path: path == CHANNELS_LIST_PATH # Only channels list exists initially
    mock_os_functions["stat"].return_value.st_size = 0 # Ensure it's treated as empty

    # Mock content of channels_to_crawl.csv
    mock_file_content = "channel_username\n@testchannel1\n@testchannel2\n"
    mock_open_func.side_effect = [
        mock_open(read_data=mock_file_content).return_value, # For reading CHANNELS_LIST_PATH
        mock_open().return_value # For writing OUTPUT_CSV_PATH (the actual CSV file)
    ]

    mock_writer_instance = mock_csv_writer_class.return_value
    
    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title="Mock Channel")
    
    # Use side_effect to provide a new AsyncMockIterator for each call to iter_messages
    # The limit (1) passed to scraper_main is passed by scrape_channel to iter_messages
    mock_telethon_client.iter_messages.side_effect = [
        lambda entity, limit: AsyncMockIterator([create_mock_message(message_text="Hello from channel 1_msg1", id=101)], limit=limit),
        lambda entity, limit: AsyncMockIterator([create_mock_message(message_text="Hello from channel 2_msg1", id=201)], limit=limit),
    ]

    await scraper_main(message_limit=1) # Limit to 1 message per channel for quick test

    mock_open_func.assert_any_call(CHANNELS_LIST_PATH, 'r', encoding='utf-8')
    mock_open_func.assert_any_call(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8')

    header_row = ['channel_title', 'channel_username', 'message_id', 'message_text',
                  'message_date', 'sender_id', 'sender_username', 'media_path', 'media_type', 'views']
    mock_writer_instance.writerow.assert_any_call(header_row)
    
    # Expected: 1 header row + 2 data rows (1 per channel, each limited to 1 message) = 3 calls
    assert mock_writer_instance.writerow.call_count == 3
    
    mock_telethon_client.start.assert_called_once()
    mock_telethon_client.disconnect.assert_called_once()
    mock_os_functions["makedirs"].assert_any_call(MEDIA_DIR, exist_ok=True)
    mock_os_functions["makedirs"].assert_any_call(RAW_DATA_DIR, exist_ok=True)


@pytest.mark.asyncio
@patch('src.data_ingestion.telegram_scraper.csv.writer') # Patch csv.writer at module level for main
@patch('src.data_ingestion.telegram_scraper.open', new_callable=mock_open) # Patch builtins.open
async def test_scraper_main_subsequent_run_appends_data(mock_open_func, mock_csv_writer_class, mock_telethon_client, mock_os_functions):
    """Tests main function when CSV file already exists and has data (subsequent run)."""
    mock_os_functions["exists"].return_value = True # Simulate file existing
    mock_os_functions["stat"].return_value.st_size = 100 # Simulate file not empty (so header is not written)

    mock_file_content = "channel_username\n@testchannel1\n"
    mock_open_func.side_effect = [
        mock_open(read_data=mock_file_content).return_value, # For reading CHANNELS_LIST_PATH
        mock_open().return_value # For writing OUTPUT_CSV_PATH (the actual CSV file)
    ]
    mock_writer_instance = mock_csv_writer_class.return_value

    mock_telethon_client.get_entity.return_value = MagicMock(spec=Channel, title="Mock Channel")
    mock_telethon_client.iter_messages.side_effect = lambda entity, limit: AsyncMockIterator([
        create_mock_message(message_text="New message from channel 1", id=301),
    ], limit=limit)

    await scraper_main(message_limit=1)

    mock_open_func.assert_any_call(CHANNELS_LIST_PATH, 'r', encoding='utf-8')
    mock_open_func.assert_any_call(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8')

    header_row = ['channel_title', 'channel_username', 'message_id', 'message_text',
                  'message_date', 'sender_id', 'sender_username', 'media_path', 'media_type', 'views']
    with pytest.raises(AssertionError): # Ensure header was NOT written
        mock_writer_instance.writerow.assert_any_call(header_row)
    
    # Expected: 1 data row (for the single channel, limited to 1 message)
    assert mock_writer_instance.writerow.call_count == 1
    mock_telethon_client.start.assert_called_once()
    mock_telethon_client.disconnect.assert_called_once()

