from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(dataset_name="wikitext", subset="wikitext-103-raw-v1", split="test"):
    """Loads the dataset from Hugging Face datasets library."""
    dataset = load_dataset(dataset_name, subset)[split]["text"]
    print(f"Loaded {len(dataset)} samples from {dataset_name} ({subset})")
    return dataset

def get_tokenizer(model_name="openai-community/gpt2"):
    """Loads a tokenizer and sets the padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer '{model_name}' loaded successfully.")
    return tokenizer

def tokenize_text(tokenizer, text_samples):
    """Tokenizes a given list of text samples."""
    tokenized_data = tokenizer(text_samples)['input_ids']
    return tokenized_data

def get_vocab_info(tokenizer, word="the"):
    """Returns the vocabulary size and token ID for a specific word."""
    vocab = tokenizer.vocab
    return len(vocab), vocab.get(word, "Not in vocabulary")

if __name__ == "__main__":
    dataset = load_data()
    tokenizer = get_tokenizer()
    sample_text = dataset[:4]
    tokenized_output = tokenize_text(tokenizer, sample_text)
    vocab_size, the_token_id = get_vocab_info(tokenizer)
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Token ID for 'the': {the_token_id}")
