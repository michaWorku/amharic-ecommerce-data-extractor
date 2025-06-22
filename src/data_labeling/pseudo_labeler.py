import re
import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path for module imports
project_root = Path(__file__).resolve().parents[2] # Adjust if script is not in tests/unit
sys.path.insert(0, str(project_root))

def tokenize_amharic_message(message: str) -> List[str]:
    """
    Basic tokenizer for Amharic text. Splits by whitespace and some punctuation.
    """
    # Normalize common symbols/punctuation
    message = message.replace('፦', ':').replace('•', '')
    # Split by spaces, and also keep some punctuation as separate tokens if needed
    tokens = re.findall(r'\b\w+\b|[.,;!?#@%/:-]', message)
    # Further split on common internal delimiters for products/prices
    cleaned_tokens = []
    for token in tokens:
        if 'ብር' in token and len(token) > 2: # e.g., "550ብር"
            parts = token.split('ብር')
            if parts[0]:
                cleaned_tokens.append(parts[0])
            cleaned_tokens.append('ብር')
        elif 'ዋጋ' in token and ':' in token: # e.g., "ዋጋ:-550"
            parts = re.split(r'[:\-]', token)
            cleaned_tokens.append(parts[0])
            cleaned_tokens.extend([p for p in parts[1:] if p])
        elif 'x' in token and (token.replace('x', '').isdigit() or any(c.isalpha() for c in token.replace('x', ''))):
            # Handle dimensions like "10meter x 45cm"
            parts = token.split('x')
            cleaned_tokens.append(parts[0])
            cleaned_tokens.append('x')
            if parts[1]:
                cleaned_tokens.append(parts[1])
        else:
            cleaned_tokens.append(token)

    # Filter out empty strings that might result from splitting
    return [t for t in cleaned_tokens if t.strip()]

def pseudo_label_message(tokens: List[str]) -> List[Dict[str, str]]:
    """
    Applies simple heuristic rules to pseudo-label tokens.
    """
    labeled_sequence = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        label = "O"

        # Rule 1: Price detection (prioritize numbers and Birr)
        if re.match(r'^\d[\d,\.]*$', token) and i + 1 < len(tokens) and tokens[i+1].lower() == 'ብር':
            label = "B-PRICE"
            labeled_sequence.append({"text": token, "label": label})
            labeled_sequence.append({"text": tokens[i+1], "label": "I-PRICE"})
            i += 2
            continue
        elif 'ብር' in token.lower() and re.search(r'\d', token): # e.g., "550ብር" or "1,200ብር"
            parts = re.split(r'(\d[\d,\.]*)', token)
            for p in parts:
                if p:
                    if re.match(r'^\d[\d,\.]*$', p):
                        if not labeled_sequence or labeled_sequence[-1]['label'] == 'O':
                             labeled_sequence.append({"text": p, "label": "B-PRICE"})
                        else: # If the previous token was part of a price, continue as I-PRICE
                            labeled_sequence.append({"text": p, "label": "I-PRICE"})
                    elif p.lower() == 'ብር':
                         labeled_sequence.append({"text": p, "label": "I-PRICE"})
                    else:
                        labeled_sequence.append({"text": p, "label": "O"})
            i += 1
            continue
        elif token.lower() in ['ዋጋ', 'ዋጋ፦', 'ዋጋ:-', 'price-']:
            label = "B-PRICE"
            labeled_sequence.append({"text": token, "label": label})
            i += 1
            if i < len(tokens):
                next_token = tokens[i]
                if re.match(r'^\d[\d,\.]*$', next_token):
                    labeled_sequence.append({"text": next_token, "label": "I-PRICE"})
                    i += 1
                    if i < len(tokens) and tokens[i].lower() == 'ብር':
                        labeled_sequence.append({"text": tokens[i], "label": "I-PRICE"})
                        i += 1
            continue

        # Rule 2: Location detection (keywords)
        location_keywords = ['መገናኛ', 'ለቡ', 'ስሪ', 'ኤም', 'ሲቲ', 'ሞል', 'ታሜ', 'ጋስ', 'ህንፃ', 'መዳህኒዓለም', 'ቤተ/ክርስቲያን', '#ዛም_ሞል', 'ቁ.1', 'ቁ.2', 'ቢሮ', 'ቁጥር']
        is_location_start = False
        for kw in location_keywords:
            if kw in token:
                is_location_start = True
                break
        
        if is_location_start:
            # Simple greedy labeling for multi-word locations
            potential_loc_tokens = []
            current_idx = i
            while current_idx < len(tokens):
                current_token = tokens[current_idx]
                if any(kw in current_token for kw in location_keywords) or \
                   re.match(r'^\d[\d,\.]*$', current_token) or \
                   current_token.lower() in ['ፎቅ', 'ፊት', 'ለ', 'ጎን', 'ቢሮ', 'ቁ.']: # Add more context words
                    potential_loc_tokens.append(current_token)
                    current_idx += 1
                else:
                    break
            
            if potential_loc_tokens:
                labeled_sequence.append({"text": potential_loc_tokens[0], "label": "B-LOC"})
                for j in range(1, len(potential_loc_tokens)):
                    labeled_sequence.append({"text": potential_loc_tokens[j], "label": "I-LOC"})
                i = current_idx
                continue

        # Rule 3: Product detection (English words, some Amharic phrases)
        # This is very basic and will need manual refinement.
        english_product_indicators = ['product', 'set', 'machine', 'maker', 'blender', 'stove', 'pan', 'humidifier', 'cleaner', 'bags', 'brush', 'lamp', 'pad', 'tape', 'mop', 'bottle', 'air fryer', 'knife', 'oven', 'grinder', 'rack', 'bowl', 'tray', 'dispenser', 'massager', 'cup', 'crepe', 'cleaner', 'towel', 'cap', 'light', 'corrector', 'blanket', 'shaper', 'heater', 'scrubber', 'epilator', 'vibrator', 'slicer', 'guard', 'warmer', 'clipper', 'lunch', 'steamer', 'play mat', 'massage', 'bag', 'diffuser', 'spice', 'container', 'hair curler', 'food storage', 'diaper', 'dumpling', 'organizer', 'charcoal burner', 'bathroom set', 'toilet', 'dish washing gloves', 'cookware', 'mixer', 'juicer', 'nutties', 'biscuits', 'omelette', 'waffles', 'snacks', 'aprons', 'abacus', 'nail', 'thermometer', 'tweezer', 'scissor', 'aspirator', 'medicine dispenser', 'tooth brush', 'epilator', 'portable', 'electric', 'stainless steel', 'silicon', 'mini', 'automatic', 'rechargeable', 'smart', 'adjustable', 'foldable', 'multipurpose', 'multinational', 'luxury', 'quality', 'style', 'expert']
        
        # Amharic product indicators (very limited and highly ambiguous)
        amharic_product_indicators = ['ምርጥ', 'እቃ', 'ልብስ', 'ማሳጅ', 'መፍጫ', 'መቁረጫ', 'ማድረቂያ', 'ማጠቢያ', 'ማስቀመጫ', 'ድስት', 'መጥበሻ', 'ቢላ', 'ሻወር', 'ዳይፐር', 'ማቅረቢያ', 'ምድጃ', 'ስቶቭ', 'ማሰሮ', 'መወልወያ', 'መብራት', 'ስብስብ', 'ስላይስ', 'ማሽን', 'ክሬም', 'ቡና', 'እንቁላል', 'ሊጥ', 'ጁስ', 'አትክልት', 'ስጋ', 'ቦርጭ', 'ፀጉር', 'ፂም', 'መቆረጪያ', 'ቶንዶስ', 'አማራጭ', 'ብርጭቆ', 'ሙቀት', 'ምጣድ', 'መጭመቂያ', 'የጽዳት', 'እንጨት', 'መደርደሪያ']

        is_product_start = False
        if token.lower() in english_product_indicators or token.lower() in amharic_product_indicators:
            is_product_start = True

        if is_product_start:
            # Simple heuristic for multi-word products
            product_phrase = [token]
            current_idx = i + 1
            while current_idx < len(tokens):
                next_token = tokens[current_idx]
                # Continue product if it's another product indicator or a descriptive adjective
                if next_token.lower() in english_product_indicators or \
                   next_token.lower() in amharic_product_indicators or \
                   re.match(r'^\d[\d,\.]*$', next_token) or \
                   next_token.lower() in ['high', 'quality', 'new', 'original', 'style', 'luxury', 'portable', 'electric', 'stainless', 'steel', 'silicon', 'mini', 'automatic', 'rechargeable', 'smart', 'adjustable', 'foldable', 'multipurpose', 'multinational', 'excellent', 'durable', 'best', 'smooth', 'effective', 'waterproof', 'ceramic', 'glass', 'bamboo', 'wooden', 'flexible', 'long', 'handled', 'dual', 'single', 'multi-function', 'set', 'pcs', 'pack', 'in']:# Common descriptors or quantity indicators
                    product_phrase.append(next_token)
                    current_idx += 1
                else:
                    break
            
            if len(product_phrase) > 0:
                labeled_sequence.append({"text": product_phrase[0], "label": "B-PRODUCT"})
                for j in range(1, len(product_phrase)):
                    labeled_sequence.append({"text": product_phrase[j], "label": "I-PRODUCT"})
                i = current_idx
                continue
        
        # Default to 'O'
        labeled_sequence.append({"text": token, "label": label})
        i += 1
    
    return labeled_sequence

def pseudo_label_file(
    input_file_path: str = 'data/labeled/messages_for_labeling.txt',
    output_file_path: str = 'data/labeled/pseudo_labeled_messages.txt'
) -> None:
    """
    Reads a text file, pseudo-labels its content, and writes it in CoNLL format.
    """
    print(f"Reading messages from: {input_file_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        messages = infile.read().strip().split('\n\n') # Split by double newline for messages

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for message in messages:
            if not message.strip():
                continue # Skip empty messages

            # Process messages that might have tokens separated by spaces or newlines
            # First, clean up extra spaces and newlines within a logical message block
            cleaned_message = ' '.join(message.split())
            
            # Tokenize the cleaned message
            tokens = tokenize_amharic_message(cleaned_message)
            
            # Pseudo-label the tokens
            labeled_tokens = pseudo_label_message(tokens)
            
            # Write to output file in CoNLL format
            for item in labeled_tokens:
                # Ensure only one space between token and label, and a newline
                outfile.write(f"{item['text']}\t{item['label']}\n")
            outfile.write('\n') # Blank line separates messages

    print(f"Pseudo-labeled data saved to: {output_file_path}")
    print("Please review this file carefully as manual corrections will be necessary for accuracy.")

if __name__ == "__main__":
    pseudo_label_file()
