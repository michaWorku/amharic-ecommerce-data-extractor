import os
import zipfile
import pandas as pd
import shutil # For cleaning up directories
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def __init__(self, extract_to_dir: str):
        """
        Initializes the ZipDataIngestor.

        Args:
            extract_to_dir (str): The directory where the zip file contents
                                  will be extracted. This directory will be created
                                  if it doesn't exist.
        """
        self.extract_to_dir = extract_to_dir
        os.makedirs(self.extract_to_dir, exist_ok=True)
        logger.info(f"Zip contents will be extracted to: {self.extract_to_dir}")

    def ingest(self, zip_file_path: str) -> pd.DataFrame:
        """
        Extracts a .zip file, reads all CSVs within it, merges them into
        a single DataFrame, and then cleans up the extracted directory.

        Args:
            zip_file_path (str): The path to the input .zip file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing data from all merged CSVs.

        Raises:
            ValueError: If the provided file is not a .zip file.
            FileNotFoundError: If no CSV files are found in the extracted data.
            Exception: For other errors during extraction or reading.
        """
        if not zip_file_path.endswith(".zip"):
            raise ValueError(f"The provided file '{zip_file_path}' is not a .zip file.")
        
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"Zip file not found at: {zip_file_path}")

        logger.info(f"Extracting zip file: {zip_file_path}...")
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.extract_to_dir)
            logger.info(f"Successfully extracted to {self.extract_to_dir}")
        except Exception as e:
            logger.error(f"Error extracting zip file '{zip_file_path}': {e}")
            raise

        # Find and read all CSV files in the extracted data directory
        extracted_csv_files = [f for f in os.listdir(self.extract_to_dir) if f.endswith(".csv")]

        if not extracted_csv_files:
            logger.error(f"No CSV files found in the extracted directory: {self.extract_to_dir}")
            raise FileNotFoundError("No CSV file found in the extracted data.")

        logger.info(f"Found {len(extracted_csv_files)} CSV files to merge.")
        
        all_dfs = []
        for csv_file_name in extracted_csv_files:
            csv_file_path = os.path.join(self.extract_to_dir, csv_file_name)
            try:
                df_temp = pd.read_csv(csv_file_path)
                all_dfs.append(df_temp)
                logger.debug(f"Read '{csv_file_name}' with {len(df_temp)} rows.")
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV file '{csv_file_name}' is empty. Skipping.")
            except Exception as e:
                logger.error(f"Error reading CSV file '{csv_file_name}': {e}")
                # Decide whether to raise or continue. For now, let's raise to be strict.
                raise

        if not all_dfs:
            logger.error("All found CSV files were empty or could not be read.")
            raise ValueError("No data could be ingested from the CSV files.")

        # Concatenate all DataFrames
        merged_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Successfully merged data from all CSVs. Total rows: {len(merged_df)}")

        # Clean up the extracted data directory
        try:
            shutil.rmtree(self.extract_to_dir)
            logger.info(f"Cleaned up extracted directory: {self.extract_to_dir}")
        except Exception as e:
            logger.warning(f"Could not remove extracted directory '{self.extract_to_dir}': {e}")

        return merged_df


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str, **kwargs) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on file extension.

        Args:
            file_extension (str): The extension of the file (e.g., ".zip").
            **kwargs: Additional arguments to pass to the ingestor constructor.

        Returns:
            DataIngestor: An instance of a concrete DataIngestor.

        Raises:
            ValueError: If no ingestor is available for the given file extension.
        """
        if file_extension.lower() == ".zip":
            # Ensure 'extract_to_dir' is provided for ZipDataIngestor
            if 'extract_to_dir' not in kwargs:
                raise ValueError("ZipDataIngestor requires 'extract_to_dir' argument.")
            return ZipDataIngestor(extract_to_dir=kwargs['extract_to_dir'])
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Example usage for testing this module directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Correct BASE_DIR calculation for nested script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(current_script_dir))

    # Define paths relative to the project root
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

    # Path for the zip file to be ingested
    # For this example, you would manually create a 'telegram_data.zip'
    # in your 'data/raw/' directory, containing one or more CSVs.
    zip_file_path = os.path.join(RAW_DATA_DIR, 'telegram_data.zip')
    
    # Directory to extract zip contents into
    extracted_data_dir = os.path.join(RAW_DATA_DIR, 'extracted_data')

    # Create a dummy zip file for testing purposes if it doesn't exist
    if not os.path.exists(zip_file_path):
        logger.info(f"Creating a dummy zip file for testing: {zip_file_path}")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        # Create some dummy CSV files inside a temporary directory
        temp_csv_dir = os.path.join(RAW_DATA_DIR, 'temp_csv_for_zip')
        os.makedirs(temp_csv_dir, exist_ok=True)

        df1 = pd.DataFrame({'colA': [1, 2], 'colB': ['x', 'y']})
        df2 = pd.DataFrame({'colA': [3, 4], 'colB': ['a', 'b']})
        
        df1.to_csv(os.path.join(temp_csv_dir, 'part1.csv'), index=False)
        df2.to_csv(os.path.join(temp_csv_dir, 'part2.csv'), index=False)

        with zipfile.ZipFile(zip_file_path, 'w') as zf:
            zf.write(os.path.join(temp_csv_dir, 'part1.csv'), 'part1.csv')
            zf.write(os.path.join(temp_csv_dir, 'part2.csv'), 'part2.csv')
        
        shutil.rmtree(temp_csv_dir) # Clean up temporary CSVs
        logger.info("Dummy zip file created.")


    try:
        # Determine the file extension
        file_extension = os.path.splitext(zip_file_path)[1]

        # Get the appropriate DataIngestor using the factory
        data_ingestor = DataIngestorFactory.get_data_ingestor(
            file_extension, 
            extract_to_dir=extracted_data_dir
        )

        # Ingest the data and load it into a DataFrame
        merged_df = data_ingestor.ingest(zip_file_path)

        # Now merged_df contains the DataFrame from the extracted and merged CSVs
        print("\nMerged DataFrame Head:")
        print(merged_df.head())
        print(f"\nMerged DataFrame Shape: {merged_df.shape}")

        # Save the merged DataFrame to a new CSV
        output_merged_csv_path = os.path.join(RAW_DATA_DIR, 'telegram_data_merged.csv')
        merged_df.to_csv(output_merged_csv_path, index=False, encoding='utf-8')
        logger.info(f"Merged data saved to: {output_merged_csv_path}")

    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}")
    finally:
        # Ensure the extracted_data_dir is cleaned up even if an error occurs
        if os.path.exists(extracted_data_dir):
            try:
                shutil.rmtree(extracted_data_dir)
                logger.info(f"Cleaned up {extracted_data_dir} in finally block.")
            except Exception as e:
                logger.warning(f"Could not remove {extracted_data_dir} in finally block: {e}")
