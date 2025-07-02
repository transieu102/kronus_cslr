import pandas as pd
from transformers import pipeline
import os
from tqdm import tqdm

def create_vocab(df, gloss_column):
    """Create vocabulary from gloss column."""
    vocab = set()
    for gloss in df[gloss_column]:
        words = gloss.split()
        vocab.update(words)
    return vocab

def process_sentence(sentence, mask_model, vocab):
    """Process a single sentence by masking each word and checking generated sentences."""
    words = sentence.split()
    valid_sentences = []
    
    for i in range(len(words)):
        # Create masked sentence
        masked_sentence = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
        
        # Generate predictions
        predictions = mask_model(masked_sentence)
        
        # Check each prediction
        for pred in predictions:
            generated_sentence = pred['sequence']
            generated_words = generated_sentence.split()
            
            # Check if all words are in vocabulary
            if all(word in vocab for word in generated_words):
                valid_sentences.append(generated_sentence)
    
    return valid_sentences

def main():
    # Initialize the mask-filling model
    print("Loading model...")
    fill_mask = pipeline(
        "fill-mask",
        model="aubmindlab/araelectra-base-generator",
        tokenizer="aubmindlab/araelectra-base-generator"
    )
    
    # Get input file path from user
    input_file = '/home/sieut/kronus/data/us/train.csv'
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = create_vocab(df, 'gloss')
    
    # Process each sentence
    print("Processing sentences...")
    synthetic_sentences = []
    
    for video_id, gloss in tqdm(df[['id', 'gloss']].values):
        valid_sentences = process_sentence(gloss, fill_mask, vocab)
        for sentence in valid_sentences:
            synthetic_sentences.append({'video_id': video_id, 'gloss': sentence})
        # synthetic_sentences.extend(valid_sentences)
    
    # Create output file path
    output_file = os.path.join(
        os.path.dirname(input_file),
        'synthetic_gloss_sentence.csv'
    )
    
    # Save results
    print(f"Saving results to {output_file}...")
    # pd.DataFrame({'synthetic_gloss': synthetic_sentences}).to_csv(output_file, index=False)
    pd.DataFrame(synthetic_sentences).to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main() 