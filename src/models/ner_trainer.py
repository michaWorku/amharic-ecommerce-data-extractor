import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Ensure src/utils is in the Python path to import conll_parser
project_root = Path(__file__).resolve().parents[2] # Adjust this if your structure differs
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from utils.conll_parser import read_conll
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_metrics(p):
    """
    Computes precision, recall, and F1-score for token classification.
    Ignores -100 labels.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [[id_to_label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    # Filter out empty lists if any (e.g. if a sentence was only special tokens after filtering)
    true_labels_filtered = [sublist for sublist in true_labels if sublist]
    true_predictions_filtered = [sublist for sublist in true_predictions if sublist]

    # Ensure true_labels and true_predictions have the same number of samples
    # This can happen if some prediction sublists are empty after filtering
    min_len = min(len(true_labels_filtered), len(true_predictions_filtered))
    true_labels_final = true_labels_filtered[:min_len]
    true_predictions_final = true_predictions_filtered[:min_len]

    if not true_labels_final or not true_predictions_final:
        logger.warning("No valid labels or predictions to compute metrics. Returning zeros.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Use "macro" average for F1, precision, and recall to account for class imbalance
    # This calculates metrics for each label and takes their unweighted mean.
    f1 = f1_score(true_labels_final, true_predictions_final, average="macro")
    precision = precision_score(true_labels_final, true_predictions_final, average="macro")
    recall = recall_score(true_labels_final, true_predictions_final, average="macro")

    logger.info("\nClassification Report:")
    logger.info(classification_report(true_labels_final, true_predictions_final))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_ner_model(
    labeled_data_path: str,
    output_model_dir: str,
    model_name: str = "xlm-roberta-base",
    train_split_ratio: float = 0.8,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
):
    """
    Fine-tunes a Named Entity Recognition (NER) model using the Hugging Face Transformers library.

    Args:
        labeled_data_path (str): Path to the CoNLL formatted labeled dataset.
        output_model_dir (str): Directory to save the fine-tuned model and tokenizer.
        model_name (str): Name of the pre-trained model from Hugging Face Hub.
        train_split_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80%).
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size per device for training.
        per_device_eval_batch_size (int): Batch size per device for evaluation.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
    """
    logger.info(f"Starting NER model training for model: {model_name}")
    logger.info(f"Loading labeled data from: {labeled_data_path}")

    # Load the labeled dataset using the utility function
    raw_data = read_conll(labeled_data_path)

    if not raw_data:
        logger.error(f"No data loaded from {labeled_data_path}. Please ensure the file exists and is correctly formatted.")
        return

    # Extract all unique labels to create ID mappings
    global id_to_label # Make id_to_label globally accessible for compute_metrics
    all_labels = sorted(list(set(item['label'] for sentence in raw_data for item in sentence)))
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    logger.info(f"Detected labels: {all_labels}")
    logger.info(f"Label to ID mapping: {label_to_id}")

    # Convert raw_data to Hugging Face Dataset format
    valid_tokens = []
    valid_ner_tags = []

    for sentence_data in raw_data:
        tokens_sentence = [item['text'] for item in sentence_data]
        ner_tags_sentence = [label_to_id[item['label']] for item in sentence_data]
        if tokens_sentence and ner_tags_sentence:
            valid_tokens.append(tokens_sentence)
            valid_ner_tags.append(ner_tags_sentence)

    hf_dataset_format = {
        "tokens": valid_tokens,
        "ner_tags": valid_ner_tags
    }

    dataset = Dataset.from_dict(hf_dataset_format)
    logger.info(f"Total samples in dataset: {len(dataset)}")

    # Split dataset into training and validation sets
    if len(dataset) < 2:
        logger.warning("Dataset has less than 2 samples. Skipping train/test split. All data used for training.")
        train_dataset = dataset
        eval_dataset = Dataset.from_dict({"tokens": [], "ner_tags": []}) # Create an empty eval dataset
    else:
        train_test_split = dataset.train_test_split(test_size=1 - train_split_ratio, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer for {model_name} loaded.")

    def align_labels_with_tokens(examples):
        """
        Function to tokenize inputs and align labels with new tokens.
        Handles potential subword tokenization by setting labels for subword pieces to -100 (ignored by PyTorch).
        """
        tokenized_inputs = tokenizer(
            examples["tokens"], is_split_into_words=True, truncation=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply tokenization and label alignment
    logger.info("Tokenizing and aligning labels...")
    tokenized_train_dataset = train_dataset.map(align_labels_with_tokens, batched=True)
    tokenized_eval_dataset = eval_dataset.map(align_labels_with_tokens, batched=True)
    logger.info("Tokenization and alignment complete.")

    # Remove original columns as they are no longer needed for training
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["tokens", "ner_tags"])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["tokens", "ner_tags"])

    # Load the pre-trained model
    logger.info(f"Loading pre-trained model: {model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(all_labels), id2label=id_to_label, label2id=label_to_id
    )
    logger.info(f"Model '{model_name}' loaded with {len(all_labels)} labels.")

    # Define Training Arguments
    os.makedirs(output_model_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch" if len(eval_dataset) > 0 else "no", # Only evaluate if eval_dataset is not empty
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True if len(eval_dataset) > 0 else False,
        metric_for_best_model="f1" if len(eval_dataset) > 0 else None,
        report_to="none",
        remove_unused_columns=False
    )
    logger.info("Training arguments defined.")

    # Initialize Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    logger.info("Data collator initialized.")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    logger.info("Trainer initialized. Starting fine-tuning...")

    # Fine-tune the model
    trainer.train()
    logger.info("Fine-tuning complete.")

    # Save the model and tokenizer
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir) # Save tokenizer with the model
    logger.info(f"Fine-tuned model and tokenizer saved to: {output_model_dir}")

    # Evaluate the model
    if len(eval_dataset) > 0:
        logger.info("Evaluating model on evaluation dataset...")
        results = trainer.evaluate()
        logger.info("\nFinal Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        return results
    else:
        logger.info("Skipping final evaluation because the evaluation dataset is empty.")
        return {}

if __name__ == '__main__':
    # Example Usage for stand-alone execution (for testing purposes)
    # This assumes you have a labeled CoNLL file at data/labeled/01_labeled_telegram_product_price_location.txt
    # You might need to create dummy data for quick testing if you don't have a labeled file yet.

    # Create dummy labeled data for testing if it doesn't exist
    LABELED_DATA_FOR_TRAINER = Path(project_root) / 'data' / 'labeled' / '01_labeled_telegram_product_price_location.txt'
    if not LABELED_DATA_FOR_TRAINER.exists():
        logger.warning(f"Labeled data not found at {LABELED_DATA_FOR_TRAINER}. Creating dummy data for demonstration.")
        dummy_data = [
            [{'text': 'Dell', 'label': 'B-PRODUCT'}, {'text': 'laptop', 'label': 'I-PRODUCT'}, {'text': 'price', 'label': 'O'}, {'text': '1000', 'label': 'B-PRICE'}, {'text': 'ETB', 'label': 'I-PRICE'}, {'text': 'at', 'label': 'O'}, {'text': 'Bole', 'label': 'B-LOC'}, {'text': 'Road', 'label': 'I-LOC'}, {'text': '.', 'label': 'O'}],
            [{'text': 'New', 'label': 'O'}, {'text': 'Phone', 'label': 'B-PRODUCT'}, {'text': 'for', 'label': 'O'}, {'text': 'sale', 'label': 'O'}, {'text': '2500', 'label': 'B-PRICE'}, {'text': 'Br', 'label': 'I-PRICE'}, {'text': 'contact', 'label': 'O'}, {'text': '+251911123456', 'label': 'B-CONTACT_INFO'}, {'text': '.', 'label': 'O'}]
        ]
        os.makedirs(LABELED_DATA_FOR_TRAINER.parent, exist_ok=True)
        write_conll(dummy_data, str(LABELED_DATA_FOR_TRAINER))

    OUTPUT_MODEL_DIR_FOR_TRAINER = Path(project_root) / 'models' / 'fine_tuned_ner_model_test'
    MODEL_NAME_FOR_TRAINER = "xlm-roberta-base" # Or "distilbert-base-multilingual-cased" for faster testing

    # Set id_to_label globally for compute_metrics in this demo scope
    # This is a temporary setup for the __main__ block. In a real pipeline,
    # id_to_label would typically be passed or loaded consistently.
    dummy_labels = ['B-CONTACT_INFO', 'B-LOC', 'B-PRICE', 'B-PRODUCT', 'I-LOC', 'I-PRICE', 'I-PRODUCT', 'O']
    id_to_label = {i: label for i, label in enumerate(sorted(dummy_labels))}

    train_ner_model(
        labeled_data_path=str(LABELED_DATA_FOR_TRAINER),
        output_model_dir=str(OUTPUT_MODEL_DIR_FOR_TRAINER),
        model_name=MODEL_NAME_FOR_TRAINER,
        num_train_epochs=1, # Keep small for quick demo
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    logger.info("NER model training demonstration finished.")
