import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import random
import torch
import logging

# Set up logging for informative messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adjust sys.path to allow imports from src
# Ensure project_root is the directory containing 'src'
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root)) # This adds the project root to path

# Corrected import: Now explicitly import from src.utils.conll_parser
from src.utils.conll_parser import read_conll # For evaluation on labeled data if needed
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, DataCollatorForTokenClassification
from datasets import Dataset # For creating Hugging Face Datasets

# Import SHAP and LIME for interpretability
try:
    import shap
    from lime.lime_text import LimeTextExplainer
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    logger.warning("SHAP or LIME not installed. Model interpretability features will be disabled.")
    INTERPRETABILITY_AVAILABLE = False
    # Define dummy classes/functions to prevent errors if not installed
    class DummyLimeTextExplainer:
        def __init__(self, class_names): pass
        def explain_instance(self, *args, **kwargs):
            logger.warning("LIME is not installed. Cannot generate explanation.")
            return None
    LimeTextExplainer = DummyLimeTextExplainer
    shap = None # Ensure shap is None if not installed


class NERModelEvaluator:
    def __init__(self, model_dir: str):
        """
        Initializes the NER Model Evaluator by loading the fine-tuned model and tokenizer.

        Args:
            model_dir (str): Directory containing the fine-tuned model and tokenizer.
        """
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.id_to_label = None
        self.label_to_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Loads the fine-tuned model, tokenizer, and sets up the NER pipeline."""
        logger.info(f"Loading fine-tuned model from: {self.model_dir}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)

            # Ensure id2label and label2id are correctly loaded from model config
            if hasattr(self.model.config, 'id2label') and hasattr(self.model.config, 'label2id'):
                self.id_to_label = self.model.config.id2label
                self.label_to_id = self.model.config.label2id
                logger.info(f"Labels loaded: {list(self.id_to_label.values())}")
            else:
                logger.error("id2label or label2id not found in model config. Cannot perform evaluation or interpretability.")
                # Attempt to infer a basic O-label if necessary, but this is a critical error
                self.id_to_label = {0: 'O'}
                self.label_to_id = {'O': 0}
                logger.warning("Defaulting to a minimal label set. This may cause errors.")

            # Specify device for pipeline explicitly
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
            logger.info("NER pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer from {self.model_dir}: {e}")
            self.tokenizer = None
            self.model = None
            self.ner_pipeline = None

    def predict_and_save_entities(self, preprocessed_data_path: str, output_predictions_csv_path: str):
        """
        Performs NER prediction on preprocessed text data (CSV with 'preprocessed_text' and 'tokens' columns)
        and saves the results to a new CSV, compatible with CoNLL-like format for manual labeling review.

        Args:
            preprocessed_data_path (str): Path to the preprocessed data CSV.
            output_predictions_csv_path (str): Path to save the CSV with predictions.
        """
        if not self.ner_pipeline:
            logger.error("NER pipeline is not initialized. Cannot perform prediction.")
            return

        logger.info(f"Loading preprocessed data from: {preprocessed_data_path}")
        if not Path(preprocessed_data_path).exists():
            logger.error(f"Preprocessed data file not found at {preprocessed_data_path}.")
            return

        try:
            df = pd.read_csv(preprocessed_data_path)
            # Ensure 'tokens' column is read as a list, not string representation of list
            if 'tokens' in df.columns:
                df['tokens'] = df['tokens'].apply(lambda x: eval(x) if pd.notna(x) and x.startswith('[') else [])
            else:
                logger.warning("'tokens' column not found. Falling back to splitting 'preprocessed_text'.")
                df['tokens'] = df['preprocessed_text'].fillna('').apply(lambda x: x.split())

            df['preprocessed_text'] = df['preprocessed_text'].fillna('') # Ensure no NaN texts

        except Exception as e:
            logger.error(f"Error loading or parsing preprocessed data: {e}")
            import traceback
            traceback.print_exc()
            return

        if df.empty:
            logger.warning("Preprocessed DataFrame is empty. Skipping prediction.")
            return

        logger.info(f"Performing NER prediction on {len(df)} messages...")
        csv_data = []
        for idx, row in df.iterrows():
            sentence_text = row['preprocessed_text']
            original_tokens = row['tokens'] # Use the 'tokens' column for better alignment

            if not sentence_text.strip(): # Skip empty or whitespace-only texts
                continue

            # Ensure sentence_text is not empty after strip() for pipeline
            if not sentence_text.strip():
                logger.debug(f"Skipping empty preprocessed_text for message_id {row.get('message_id', idx)}")
                continue

            prediction = self.ner_pipeline(sentence_text)

            # Align predicted entities with original tokens
            predicted_labels_aligned = ["O"] * len(original_tokens)
            
            # Simple greedy alignment: find which tokens overlap with predicted spans
            # This can be refined for more precise token-level alignment if needed.
            char_to_token_map = []
            current_char_idx = 0
            for token_idx, token in enumerate(original_tokens):
                # Ensure the token has a valid starting position in the sentence_text
                # If there are multiple spaces, find() might be tricky. Using regex might be more robust
                # For simplicity now, assume single space separation from tokenization
                token_start_in_sentence = sentence_text.find(token, current_char_idx)
                if token_start_in_sentence == -1: # Fallback if token not found from current_char_idx
                    token_start_in_sentence = sentence_text.find(token)
                    if token_start_in_sentence == -1: # If still not found, skip this token in char map
                        continue
                
                token_end_in_sentence = token_start_in_sentence + len(token)
                for char_idx in range(token_start_in_sentence, token_end_in_sentence):
                    char_to_token_map.append(token_idx)
                
                # Advance current_char_idx past this token and any following space
                current_char_idx = token_end_in_sentence
                if current_char_idx < len(sentence_text) and sentence_text[current_char_idx] == ' ':
                    current_char_idx += 1 # Account for the space

            for pred in prediction:
                pred_start = pred['start']
                pred_end = pred['end']
                pred_label = pred['entity_group']

                # Find tokens that fully or partially overlap with the prediction span
                for char_idx in range(pred_start, pred_end):
                    if char_idx < len(char_to_token_map):
                        token_idx = char_to_token_map[char_idx]
                        # Assign B- tag to the first token of the entity, I- to subsequent
                        if pred['start'] <= char_idx < pred['end']: # Ensure char_idx is within prediction span
                            if predicted_labels_aligned[token_idx] == "O": # Only assign if not already set by a prior (overlapping) prediction
                                if char_idx == pred_start or (token_idx > 0 and char_to_token_map[char_idx-1] != token_idx): # Check if it's the start of the entity or a new token in the entity
                                    predicted_labels_aligned[token_idx] = f"B-{pred_label}"
                                else:
                                    predicted_labels_aligned[token_idx] = f"I-{pred_label}"
                            elif predicted_labels_aligned[token_idx].startswith("B-") and pred_label in predicted_labels_aligned[token_idx]:
                                # If already a B- for the same entity type, it means this token is part of a multi-token B-
                                pass # Keep the B-
                            elif predicted_labels_aligned[token_idx].startswith("I-") and pred_label in predicted_labels_aligned[token_idx]:
                                # If already an I- for the same entity type
                                pass # Keep the I-
                            else: # If existing label is different (e.g., O or another entity type), prioritize the new prediction
                                if char_idx == pred_start or (token_idx > 0 and char_to_token_map[char_idx-1] != token_idx): # Check if it's the start of the entity or a new token in the entity
                                    predicted_labels_aligned[token_idx] = f"B-{pred_label}"
                                else:
                                    predicted_labels_aligned[token_idx] = f"I-{pred_label}"
                

            # Iterate through original tokens to write to CSV
            for i, token in enumerate(original_tokens):
                csv_data.append({
                    'message_id': row.get('message_id', idx), # Use actual message_id if available
                    'channel_username': row.get('channel_username', 'N/A'),
                    'token': token,
                    'predicted_label': predicted_labels_aligned[i] if i < len(predicted_labels_aligned) else "O",
                    'preprocessed_text': sentence_text # Keep original sentence for context
                })
            # Add a blank row after each sentence for readability during manual review
            csv_data.append({
                'message_id': row.get('message_id', idx),
                'channel_username': row.get('channel_username', 'N/A'),
                'token': '',
                'predicted_label': '',
                'preprocessed_text': ''
            })

        # Remove the last blank row if present (prevents an extra blank line at EOF)
        if csv_data and csv_data[-1]['token'] == '' and csv_data[-1]['predicted_label'] == '':
            csv_data.pop()

        predictions_df = pd.DataFrame(csv_data)
        # Ensure output directory exists before saving
        os.makedirs(Path(output_predictions_csv_path).parent, exist_ok=True)
        predictions_df.to_csv(output_predictions_csv_path, index=False)
        logger.info(f"Predicted labels saved to {output_predictions_csv_path}")
        logger.info("You can now manually review and correct this CSV to create more labeled data.")

    def analyze_interpretability(self, eval_dataset: Dataset, num_examples: int = 5):
        """
        Performs SHAP and LIME interpretability analysis on selected examples from the evaluation dataset.

        Args:
            eval_dataset (Dataset): The evaluation dataset in Hugging Face Dataset format.
            num_examples (int): Number of examples to select for interpretability analysis.
        """
        if not INTERPRETABILITY_AVAILABLE:
            logger.warning("Skipping interpretability analysis as SHAP or LIME is not installed.")
            return

        if not self.ner_pipeline or not self.tokenizer or not self.model or not self.id_to_label or not self.label_to_id:
            logger.error("Model, tokenizer, or label mappings not initialized. Cannot perform interpretability.")
            return

        if len(eval_dataset) == 0:
            logger.warning("Evaluation dataset is empty. Skipping interpretability analysis.")
            return

        logger.info(f"Selecting {num_examples} examples for interpretability analysis.")
        random.seed(42) # Set seed for reproducibility

        selected_indices = random.sample(range(len(eval_dataset)), min(num_examples, len(eval_dataset)))
        selected_examples = [eval_dataset[i] for i in selected_indices]

        analysis_data = []
        logger.info("Getting predictions and aligning labels for interpretability examples...")
        for example_index, example in enumerate(selected_examples):
            original_tokens = example['tokens']
            true_labels_ids = example['ner_tags']
            true_labels = [self.id_to_label[tag_id] for tag_id in true_labels_ids]
            sentence_text = " ".join(original_tokens)

            # Skip empty sentences
            if not sentence_text.strip():
                continue

            prediction = self.ner_pipeline(sentence_text)

            # Align predicted entities with original tokens
            predicted_labels_aligned = ["O"] * len(original_tokens)
            char_to_token_map = []
            current_char_idx = 0
            for token_idx, token in enumerate(original_tokens):
                token_start_in_sentence = sentence_text.find(token, current_char_idx)
                if token_start_in_sentence == -1:
                    token_start_in_sentence = sentence_text.find(token)
                    if token_start_in_sentence == -1:
                        continue
                token_end_in_sentence = token_start_in_sentence + len(token)
                for char_idx in range(token_start_in_sentence, token_end_in_sentence):
                    char_to_token_map.append(token_idx)
                current_char_idx = token_end_in_sentence
                if current_char_idx < len(sentence_text) and sentence_text[current_char_idx] == ' ':
                    current_char_idx += 1

            for pred in prediction:
                pred_start = pred['start']
                pred_end = pred['end']
                pred_label = pred['entity_group']

                for char_idx in range(pred_start, pred_end):
                    if char_idx < len(char_to_token_map):
                        token_idx = char_to_token_map[char_idx]
                        if char_idx == pred_start or (token_idx > 0 and char_to_token_map[char_idx-1] != token_idx):
                            predicted_labels_aligned[token_idx] = f"B-{pred_label}"
                        else:
                            predicted_labels_aligned[token_idx] = f"I-{pred_label}"


            mismatches = []
            has_mismatch = False
            for i in range(len(original_tokens)):
                if true_labels[i] != predicted_labels_aligned[i]:
                    mismatch_type = "Misclassification" if true_labels[i] != "O" and predicted_labels_aligned[i] != "O" else ("False Positive" if true_labels[i] == "O" else "False Negative")
                    mismatches.append({
                        'token': original_tokens[i],
                        'true_label': true_labels[i],
                        'predicted_label': predicted_labels_aligned[i],
                        'type': mismatch_type
                    })
                    has_mismatch = True

            analysis_data.append({
                'example_index': selected_indices[example_index],
                'original_tokens': original_tokens,
                'sentence_text': sentence_text, # Add full sentence for LIME
                'true_labels': true_labels,
                'predicted_labels_aligned': predicted_labels_aligned,
                'has_mismatch': has_mismatch,
                'mismatches': mismatches
            })
        logger.info("Prediction and alignment for interpretability examples complete.")

        difficult_cases = [ex for ex in analysis_data if ex['has_mismatch']]
        correct_cases = [ex for ex in analysis_data if not ex['has_mismatch']]

        logger.info(f"Selected {len(difficult_cases)} difficult cases and {len(correct_cases)} correct cases for detailed analysis.")

        # --- SHAP Analysis ---
        self._run_shap_analysis(difficult_cases, correct_cases)

        # --- LIME Analysis ---
        self._run_lime_analysis(difficult_cases, correct_cases)

    def _run_shap_analysis(self, difficult_cases: List[Dict], correct_cases: List[Dict]):
        """Internal method to perform SHAP analysis."""
        if not shap:
            return

        logger.info("\n--- Starting SHAP Analysis ---")

        # Prepare a background dataset for KernelSHAP (small sample from training or general text)
        # For a practical setup, this should be from a representative dataset.
        # Here, we'll create a dummy one for demonstration.
        dummy_background_tokens = [
            ["ይህ", "ምርት", "በጣም", "ጥሩ", "ነው", "።"],
            ["አዲስ", "ላፕቶፕ", "ዋጋው", "ከፍተኛ", "ነው", "።"],
            ["ቦሌ", "አካባቢ", "ትልቅ", "ሱቅ", "አለ", "።"]
        ]
        # Tokenize and format for SHAP background
        background_dataset = Dataset.from_dict({"tokens": dummy_background_tokens, "ner_tags": [[0]*len(t) for t in dummy_background_tokens]}) # Dummy NER tags
        tokenized_background = background_dataset.map(self._align_labels_with_tokens_for_shap, batched=True, remove_columns=["tokens", "ner_tags"])

        num_background_samples = min(10, len(tokenized_background)) # Use a very small sample for quick demo
        if num_background_samples == 0:
            logger.warning("No background samples for SHAP. Skipping SHAP analysis.")
            return

        background_indices = random.sample(range(len(tokenized_background)), num_background_samples)
        background_dataset_raw = [tokenized_background[i] for i in background_indices]

        # Use DataCollatorForTokenClassification
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        background_dict_padded = data_collator(background_dataset_raw)

        if not background_dict_padded['input_ids'].numel(): # Check if tensor is empty
            logger.warning("Empty background dataset after collation. Skipping SHAP.")
            return

        # Stack input_ids and attention_mask for SHAP explainer input
        background_input_ids_np_shap = background_dict_padded['input_ids'].cpu().numpy()
        background_attention_mask_np_shap = background_dict_padded['attention_mask'].cpu().numpy()
        background_data_stacked = np.stack([background_input_ids_np_shap, background_attention_mask_np_shap], axis=1)

        examples_for_shap = difficult_cases[:1] + correct_cases[:1] # Take 1 of each for brevity

        if not examples_for_shap:
            logger.info("No suitable examples for SHAP analysis.")
            return

        for example_count, example_case in enumerate(examples_for_shap):
            logger.info(f"--- SHAP: Analyzing Example {example_count + 1} (Original Index: {example_case['example_index']}) ---")
            original_tokens = example_case['original_tokens']
            sentence_text = example_case['sentence_text']

            tokenized_output_for_word_ids = self.tokenizer(
                [original_tokens], is_split_into_words=True, truncation=True, padding='do_not_pad'
            )

            if not tokenized_output_for_word_ids['input_ids'] or not tokenized_output_for_word_ids['input_ids'][0]:
                logger.warning(f"Could not tokenize original tokens for example {example_case['example_index']}. Skipping SHAP.")
                continue

            original_word_ids = tokenized_output_for_word_ids.word_ids(batch_index=0)
            input_ids_raw = tokenized_output_for_word_ids['input_ids'][0]
            attention_mask_raw = tokenized_output_for_word_ids['attention_mask'][0]

            example_dict_for_collation = {
                'input_ids': input_ids_raw,
                'attention_mask': attention_mask_raw,
                'labels': [-100] * len(input_ids_raw) # Labels needed by data collator, but -100 means ignore
            }
            example_dict_padded = data_collator([example_dict_for_collation])
            input_ids_padded = example_dict_padded['input_ids'][0].tolist()
            attention_mask_padded = example_dict_padded['attention_mask'][0].tolist()
            tokenized_tokens_padded = self.tokenizer.convert_ids_to_tokens(input_ids_padded)

            # Select target token and label for explanation
            target_tokenized_index = None
            target_original_token_index = None
            target_label_name = "O" # Default

            # Find a target (e.g., the first predicted entity that is not 'O')
            for i, label in enumerate(example_case['predicted_labels_aligned']):
                if label != "O":
                    target_original_token_index = i
                    target_label_name = label
                    # Find corresponding tokenized index
                    for tok_idx, word_idx in enumerate(original_word_ids):
                        if word_idx == target_original_token_index:
                            target_tokenized_index = tok_idx
                            break
                    if target_tokenized_index is not None:
                        break
            if target_tokenized_index is None and original_tokens: # Fallback to first token
                target_original_token_index = 0
                target_label_name = example_case['predicted_labels_aligned'][0] if example_case['predicted_labels_aligned'] else "O"
                for tok_idx, word_idx in enumerate(original_word_ids):
                        if word_idx == target_original_token_index:
                            target_tokenized_index = tok_idx
                            break

            if target_tokenized_index is None or target_label_name not in self.label_to_id:
                logger.warning(f"Could not determine a suitable target for SHAP for example {example_case['example_index']}. Skipping.")
                continue

            target_label_id = self.label_to_id[target_label_name]

            # Define predict_fn for SHAP
            def predict_fn_for_shap(inputs_np):
                input_ids_np = inputs_np[:, 0, :]
                attention_mask_np = inputs_np[:, 1, :]
                input_ids_tensor = torch.tensor(input_ids_np, dtype=torch.long, device=self.device)
                attention_mask_tensor = torch.tensor(attention_mask_np, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
                logits = outputs.logits.detach().cpu().numpy()

                # Ensure target_tokenized_index is within bounds of the logits for the current batch
                if target_tokenized_index >= logits.shape[1]:
                    # This can happen if the sequence length changes due to padding/truncation
                    # For SHAP, the input sequence length is fixed by the background dataset's padding.
                    # This condition might indicate an issue with how target_tokenized_index is determined,
                    # or an edge case with very short sequences where the target token is truncated.
                    logger.warning(f"target_tokenized_index ({target_tokenized_index}) out of bounds for batch shape {logits.shape}. Returning zeros.")
                    return np.zeros((logits.shape[0], len(self.id_to_label))) # Return probabilities for all labels
                
                # SHAP needs probabilities for all classes at the target token position
                # Assuming outputs.logits is [batch_size, sequence_length, num_labels]
                # Apply softmax to get probabilities
                probabilities = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
                return probabilities[:, target_tokenized_index, :] # Return probabilities for all labels at the target token

            # Initialize Explainer, specifying output type if model outputs logits
            # For KernelExplainer, the model_to_explain needs to output class probabilities or raw outputs
            # We adjusted predict_fn_for_shap to return probabilities for all labels for target token.
            explainer = shap.KernelExplainer(predict_fn_for_shap, background_data_stacked)
            logger.info("Calculating SHAP values (this may take some time)...")
            try:
                example_data_stacked = np.stack([np.array(input_ids_padded), np.array(attention_mask_padded)], axis=0)[np.newaxis, :, :]
                # shap_values will be a list of arrays, one for each output class
                # shap_values[target_label_id] gives values for the target label
                shap_values = explainer.shap_values(example_data_stacked)
                
                # Check if shap_values is a list (for multi-output models) and get the relevant one
                if isinstance(shap_values, list):
                    shap_values_for_target_label = shap_values[target_label_id]
                else:
                    # If it's a single array, it implies a single-output model or explainer configured differently
                    shap_values_for_target_label = shap_values
                
                shap_values_flat = shap_values_for_target_label[0, :] # Extract values for the single example

                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray): # For multi-output, expected_value is array
                    base_value = base_value[target_label_id] # Get base value for the specific target label

                logger.info("SHAP calculation complete.")

                logger.info(f"\nSHAP Visualization for Example {example_count + 1} (Original Index: {example_case['example_index']}):")
                shap.initjs() # Initialize JavaScript for SHAP plots
                display(shap.force_plot(base_value, shap_values_flat, tokenized_tokens_padded, matplotlib=False, show=False))
                # shap.waterfall_plot is also good for single examples
                # display(shap.waterfall_plot(shap.Explanation(values=shap_values_flat, base_values=base_value, data=tokenized_tokens_padded, feature_names=tokenized_tokens_padded)))

                logger.info("Interpretation:")
                logger.info(f"- Explaining prediction for token '{original_tokens[target_original_token_index]}' towards label '{target_label_name}'.")
                logger.info("- Red indicates positive contribution, blue indicates negative contribution.")
                logger.info("- Analyze which tokens influence the target label's prediction.")

            except Exception as e:
                logger.error(f"Error calculating or visualizing SHAP values for example {example_case['example_index']}: {e}")
                import traceback
                traceback.print_exc()

        logger.info("SHAP analysis complete.")

    def _run_lime_analysis(self, difficult_cases: List[Dict], correct_cases: List[Dict]):
        """Internal method to perform LIME analysis."""
        if not INTERPRETABILITY_AVAILABLE:
            return

        logger.info("\n--- Starting LIME Analysis ---")
        class_names = list(self.id_to_label.values()) # Use values from id_to_label
        explainer = LimeTextExplainer(class_names=class_names)

        def predictor_for_target_token(texts: list, target_original_token_index: int):
            probabilities_list = []
            for text in texts:
                tokenized_input = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    is_split_into_words=False
                ).to(self.device)

                word_ids = tokenized_input.word_ids(batch_index=0)
                target_tokenized_index = None
                if word_ids:
                    # Find the corresponding tokenized index for the target original token
                    for tokenized_idx, word_idx in enumerate(word_ids):
                         if word_idx is not None and word_idx == target_original_token_index:
                             target_tokenized_index = tokenized_idx
                             break

                if target_tokenized_index is not None and target_tokenized_index < tokenized_input['input_ids'].shape[1]:
                    with torch.no_grad():
                        outputs = self.model(**tokenized_input)
                    probabilities = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                    target_token_probabilities = probabilities[0, target_tokenized_index, :]
                    probabilities_list.append(target_token_probabilities)
                else:
                    num_labels = len(class_names)
                    probabilities_list.append(np.ones(num_labels) / num_labels) # Uniform distribution as a fallback

            return np.array(probabilities_list)

        examples_for_lime = difficult_cases[:1] + correct_cases[:1] # Take 1 of each for brevity

        if not examples_for_lime:
            logger.info("No suitable examples for LIME analysis.")
            return

        for example_count, example_case in enumerate(examples_for_lime):
            logger.info(f"--- LIME: Analyzing Example {example_count + 1} (Original Index: {example_case['example_index']}) ---")
            original_tokens = example_case['original_tokens']
            sentence_text = example_case['sentence_text']
            predicted_labels = example_case['predicted_labels_aligned']

            target_original_token_index = None
            target_label_name = None

            # Prioritize first mismatch for LIME
            if example_case['has_mismatch'] and example_case['mismatches']:
                first_mismatch = example_case['mismatches'][0]
                try:
                    target_original_token_index = original_tokens.index(first_mismatch['token'])
                    target_label_name = first_mismatch['predicted_label']
                except ValueError:
                    logger.warning(f"Mismatch token '{first_mismatch['token']}' not found in original_tokens. Fallback.")

            # Fallback to first non-'O' predicted entity
            if target_original_token_index is None or target_label_name not in self.label_to_id:
                for original_token_idx, pred_label in enumerate(predicted_labels):
                    if pred_label != "O" and pred_label in self.label_to_id:
                        target_original_token_index = original_token_idx
                        target_label_name = pred_label
                        break

            # Final fallback to the very first token
            if target_original_token_index is None or target_label_name not in self.label_to_id:
                if original_tokens:
                    target_original_token_index = 0
                    target_label_name = predicted_labels[0] if predicted_labels and predicted_labels[0] in self.label_to_id else "O"
                    if target_label_name not in self.label_to_id:
                        target_label_name = "O"
                else:
                    logger.warning(f"Example {example_case['example_index']} is empty. Skipping LIME.")
                    continue

            # Ensure target_label_name is in class_names and get its index
            if target_label_name not in class_names:
                logger.warning(f"Target label '{target_label_name}' not in class names. Falling back to 'O' label for LIME.")
                target_label_name = "O"
            
            target_label_id_for_lime = class_names.index(target_label_name) # Get index from class_names list

            logger.info(f"Explaining prediction for token '{original_tokens[target_original_token_index]}' towards label '{target_label_name}' (ID: {target_label_id_for_lime}).")

            try:
                explanation = explainer.explain_instance(
                    text_instance=sentence_text,
                    classifier_fn=lambda texts: predictor_for_target_token(texts, target_original_token_index),
                    labels=[target_label_id_for_lime],
                    num_features=10,
                )
                logger.info(f"\nLIME Visualization for Example {example_count + 1}:")
                # LIME visualizations are typically interactive HTML. For a script, you might want to save them.
                # For now, we'll just log that it's being generated.
                logger.info("LIME explanation generated (interactive visualization typically shown in notebooks).")
                # If you are running this in an environment that can display HTML:
                # from IPython.display import display, HTML
                # display(HTML(explanation.as_html()))


                logger.info("Interpretation:")
                logger.info(f"- The words highlighted in green contribute positively to the prediction of '{target_label_name}' for the target token.")
                logger.info(f"- Words in red contribute negatively.")

            except Exception as e:
                logger.error(f"Error generating LIME explanation for example {example_case['example_index']}: {e}")
                import traceback
                traceback.print_exc()

        logger.info("LIME analysis complete.")


    def _align_labels_with_tokens_for_shap(self, examples):
        """Helper for tokenizing and aligning labels for SHAP background dataset."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"], is_split_into_words=True, truncation=True
        )
        tokenized_inputs["labels"] = [[-100] * len(ids) for ids in tokenized_inputs["input_ids"]]
        return tokenized_inputs

if __name__ == '__main__':
    # Example Usage for stand-alone execution (for testing purposes)
    MODEL_DIR_FOR_EVALUATOR = Path(project_root) / 'models' / 'fine_tuned_ner_model_test' # Ensure this points to a trained model
    PREPROCESSED_DATA_FOR_PREDICTION = Path(project_root) / 'data' / 'processed' / 'preprocessed_telegram_data.csv'
    OUTPUT_PREDICTIONS_FOR_EVALUATOR = Path(project_root) / 'data' / 'predicted' / 'predicted_data_for_labeling_test.csv'
    LABELED_DATA_FOR_EVAL_FOR_INTERP = Path(project_root) / 'data' / 'labeled' / '01_labeled_telegram_product_price_location.txt'


    # Create dummy preprocessed data if it doesn't exist for prediction demo
    if not PREPROCESSED_DATA_FOR_PREDICTION.exists():
        logger.warning(f"Preprocessed data not found at {PREPROCESSED_DATA_FOR_PREDICTION}. Creating dummy data for prediction demo.")
        dummy_df_data = {
            'message_id': [1, 2, 3],
            'channel_username': ['test_channel_A', 'test_channel_B', 'test_channel_A'],
            'preprocessed_text': [
                "Dell laptop with 16GB RAM for sale at Bole road price 25000 ETB contact +251912345678",
                "New Phone for sale 2500 Br contact +251911123456",
                "Great watch for 500 birr available at Mexico"
            ],
            'tokens': [
                "['Dell', 'laptop', 'with', '16GB', 'RAM', 'for', 'sale', 'at', 'Bole', 'road', 'price', '25000', 'ETB', 'contact', '+251912345678']",
                "['New', 'Phone', 'for', 'sale', '2500', 'Br', 'contact', '+251911123456']",
                "['Great', 'watch', 'for', '500', 'birr', 'available', 'at', 'Mexico']"
            ]
        }
        dummy_df = pd.DataFrame(dummy_df_data)
        os.makedirs(PREPROCESSED_DATA_FOR_PREDICTION.parent, exist_ok=True)
        dummy_df.to_csv(PREPROCESSED_DATA_FOR_PREDICTION, index=False)

    # Create dummy labeled data if it doesn't exist for interpretability demo
    if not LABELED_DATA_FOR_EVAL_FOR_INTERP.exists():
        logger.warning(f"Labeled data for interpretability demo not found at {LABELED_DATA_FOR_EVAL_FOR_INTERP}. Creating dummy data.")
        dummy_labeled_data = [
            [{'text': 'Dell', 'label': 'B-PRODUCT'}, {'text': 'laptop', 'label': 'I-PRODUCT'}, {'text': 'price', 'label': 'O'}, {'text': '1000', 'label': 'B-PRICE'}, {'text': 'ETB', 'label': 'I-PRICE'}, {'text': 'at', 'label': 'O'}, {'text': 'Bole', 'label': 'B-LOC'}, {'text': 'Road', 'label': 'I-LOC'}, {'text': '.', 'label': 'O'}],
            [{'text': 'New', 'label': 'O'}, {'text': 'Phone', 'label': 'B-PRODUCT'}, {'text': 'for', 'label': 'O'}, {'text': 'sale', 'label': 'O'}, {'text': '2500', 'label': 'B-PRICE'}, {'text': 'Br', 'label': 'I-PRICE'}, {'text': 'contact', 'label': 'O'}, {'text': '+251911123456', 'label': 'B-CONTACT_INFO'}, {'text': '.', 'label': 'O'}]
        ]
        os.makedirs(LABELED_DATA_FOR_EVAL_FOR_INTERP.parent, exist_ok=True)
        # Use write_conll from src.utils.conll_parser for consistency
        from src.utils.conll_parser import write_conll as write_conll_for_demo
        write_conll_for_demo(dummy_labeled_data, str(LABELED_DATA_FOR_EVAL_FOR_INTERP))

    # Initialize evaluator
    evaluator = NERModelEvaluator(str(MODEL_DIR_FOR_EVALUATOR))

    if evaluator.ner_pipeline: # Check if model loaded successfully
        # 1. Predict and Save Entities
        evaluator.predict_and_save_entities(
            preprocessed_data_path=str(PREPROCESSED_DATA_FOR_PREDICTION),
            output_predictions_csv_path=str(OUTPUT_PREDICTIONS_FOR_EVALUATOR)
        )

        # 2. Analyze Interpretability (requires labeled data for evaluation set creation)
        if LABELED_DATA_FOR_EVAL_FOR_INTERP.exists():
            # Re-read the labeled data for creating evaluation dataset for interpretability
            raw_data_interp = read_conll(str(LABELED_DATA_FOR_EVAL_FOR_INTERP))
            if raw_data_interp:
                all_labels_interp = sorted(list(set(item['label'] for sentence in raw_data_interp for item in sentence)))
                label_to_id_interp = {label: i for i, label in enumerate(all_labels_interp)}
                id_to_label_interp = {i: label for label, i in label_to_id_interp.items()}

                valid_tokens_interp = []
                valid_ner_tags_interp = []
                for sentence_data in raw_data_interp:
                    tokens_sentence = [item['text'] for item in sentence_data]
                    ner_tags_sentence = [label_to_id_interp[item['label']] for item in sentence_data]
                    if tokens_sentence and ner_tags_sentence:
                        valid_tokens_interp.append(tokens_sentence)
                        valid_ner_tags_interp.append(ner_tags_sentence)
                hf_dataset_format_interp = {"tokens": valid_tokens_interp, "ner_tags": valid_ner_tags_interp}
                eval_dataset_for_interp = Dataset.from_dict(hf_dataset_format_interp)

                evaluator.id_to_label = id_to_label_interp # Ensure evaluator has correct mappings
                evaluator.label_to_id = label_to_id_interp

                evaluator.analyze_interpretability(eval_dataset=eval_dataset_for_interp, num_examples=2)
            else:
                logger.warning("No data found for interpretability analysis after reading labeled data.")
        else:
            logger.warning("Skipping interpretability analysis demo as labeled data for evaluation set is missing.")
    else:
        logger.error("Evaluator could not be initialized, likely due to missing model files. Skipping all evaluation tasks.")

    logger.info("NER model evaluation demonstration finished.")
