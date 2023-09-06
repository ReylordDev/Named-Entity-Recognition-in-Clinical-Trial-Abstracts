import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

# Global variables
TRAIN_DATA_PATH = "data/train.json"
TEST_DATA_PATH = "data/test.json"
PAD = "<PAD>"


# Preprocessing Functions
def load_data(train_file_path, test_file_path):
    """
    Load JSON data from the given file paths.

    Parameters:
    train_file_path (str): Path to the training JSON file.
    test_file_path (str): Path to the testing JSON file.

    Returns:
    tuple: Loaded JSON data for training and testing.
    """
    with open(train_file_path, "r") as f:
        train_data = json.load(f)
    with open(test_file_path, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


def extract_sentences_and_labels(data):
    """
    Extract sentences and raw labels from the loaded JSON data.

    Parameters:
    data (list): Loaded JSON data.

    Returns:
    tuple: Lists of sentences and corresponding raw label data.
    """
    sentences = []
    labels = []
    for entry in data:
        for sentence in entry["sentences"]:
            sentences.append(sentence["words"])
            labels.append(sentence["entities"])
    return sentences, labels


def generate_label_vocab(labels):
    """
    Generate a list of unique labels from the raw label data.

    Parameters:
    labels (list): Raw label data.

    Returns:
    list: A list of unique labels found in the dataset.
    """
    unique_labels = set()
    for sentence_labels in labels:
        for label in sentence_labels:
            unique_labels.add(label["label"])
    return list(unique_labels)


def encode_sentences(sentences, word_to_idx):
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentence = [
            word_to_idx.get(word, word_to_idx.get("<UNK>")) for word in sentence
        ]
        encoded_sentences.append(encoded_sentence)
    return encoded_sentences


def build_word_to_idx(sentences):
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return word_to_idx


def build_idx_to_word(word_to_idx):
    return {idx: word for word, idx in word_to_idx.items()}


def build_label_to_idx(label_vocab):
    return {label: idx for idx, label in enumerate(label_vocab)}


def build_idx_to_label(label_to_idx):
    return {idx: label for label, idx in label_to_idx.items()}


def encode_labels(labels, label_vocab, sentences):
    """
    Convert the raw label data to a multi-hot encoded format.

    Parameters:
    labels (list): Raw label data.
    label_vocab (list): Vocabulary of unique labels.

    Returns:
    list: Multi-hot encoded labels for each token in the sentences.
    """
    encoded_labels = []
    for sentence_labels, sentence in zip(labels, sentences):
        sentence_encoded = []
        for i in range(len(sentence)):  # Enumerate over tokens in the sentence
            token_encoded = [0] * len(label_vocab)  # Initialize with zeros
            for label_span in sentence_labels:
                # Check if the current token is within the label span
                if i >= label_span["start_pos"] and i <= label_span["end_pos"]:
                    index = label_vocab.index(label_span["label"])
                    token_encoded[index] = 1
            sentence_encoded.append(token_encoded)
        encoded_labels.append(sentence_encoded)
    return encoded_labels


def pad_sequences(sequences, pad_value, max_length):
    """
    Pad the tokenized sentences and labels to a fixed length.

    Parameters:
    sequences (list): List of tokenized sentences or labels.
    pad_value (str or int): Value to use for padding.
    max_length (int): Maximum length to pad the sequences to.

    Returns:
    list: Padded sentences or labels.
    """
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) >= max_length:
            padded_sequences.append(sequence[:max_length])
        else:
            padded_sequence = sequence + [pad_value] * (max_length - len(sequence))
            padded_sequences.append(padded_sequence)
    return padded_sequences


def find_max_length(sentences):
    """
    Find the maximum length among all sentences.

    Parameters:
    sentences (list): List of tokenized sentences.

    Returns:
    int: Maximum length among all sentences.
    """
    return max(len(sentence) for sentence in sentences)


class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": torch.tensor(self.sentences[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def split_data(sentences, labels, split_ratio=0.9):
    """
    Split the data into training and validation sets
    """
    # Determine the split index
    split_idx = int(len(sentences) * split_ratio)

    # Split the sentences and labels
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]

    return train_sentences, train_labels, val_sentences, val_labels


def create_data_loaders(
    train_sentences,
    train_labels,
    val_sentences,
    val_labels,
    test_sentences,
    test_labels,
    batch_size=32,
):
    """
    Create DataLoaders for training, validation, and test datasets
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    # Create training, validation, and test datasets
    train_dataset = NERDataset(train_sentences, train_labels)
    val_dataset = NERDataset(val_sentences, val_labels)
    test_dataset = NERDataset(test_sentences, test_labels)

    # Create DataLoaders for each dataset
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size, generator=generator
    )

    return train_loader, val_loader, test_loader


# Utility Functions
def sample_data(sentences, labels, sample_size=5):
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    sampled_sentences, sampled_labels = zip(*random.sample(combined, sample_size))
    return list(sampled_sentences), list(sampled_labels)


def tensor_to_sentences(tensor, idx_to_word):
    decoded_sentences = []
    for sentence in tensor:
        decoded_sentence = [idx_to_word[idx.item()] for idx in sentence]
        decoded_sentences.append((decoded_sentence))
    return decoded_sentences


def tensor_to_labels(tensor, idx_to_label):
    decoded_labels = []
    for sentence_labels in tensor:
        decoded_sentence_labels = []
        for token_labels in sentence_labels:
            token_decoded_labels = [
                idx_to_label[idx]
                for idx, value in enumerate(token_labels)
                if value == 1
            ]
            decoded_sentence_labels.append(token_decoded_labels)
        decoded_labels.append(decoded_sentence_labels)
    return decoded_labels


def prepare_data_pipeline(
    train_file_path=TRAIN_DATA_PATH, test_file_path=TEST_DATA_PATH, debug=False
):
    # Load data
    train_data, test_data = load_data(train_file_path, test_file_path)

    # Extract sentences and labels
    train_sentences, train_raw_labels = extract_sentences_and_labels(train_data)
    test_sentences, test_raw_labels = extract_sentences_and_labels(test_data)

    # Generate label vocabulary
    label_vocab = generate_label_vocab(train_raw_labels + test_raw_labels)

    # Encode labels
    train_encoded_labels = encode_labels(train_raw_labels, label_vocab, train_sentences)
    test_encoded_labels = encode_labels(test_raw_labels, label_vocab, test_sentences)

    # Build the conversion dictionaries
    word_to_idx = build_word_to_idx(train_sentences + test_sentences)
    idx_to_word = build_idx_to_word(word_to_idx)
    label_to_idx = build_label_to_idx(label_vocab)
    idx_to_label = build_idx_to_label(label_to_idx)

    # Encode sentences
    train_encoded_sentences = encode_sentences(train_sentences, word_to_idx)
    test_encoded_sentences = encode_sentences(test_sentences, word_to_idx)

    # Find max length for dynamic padding
    MAX_LENGTH = find_max_length(train_sentences + test_sentences)

    # Pad sequences
    padded_train_sentences = pad_sequences(train_encoded_sentences, 0, MAX_LENGTH)
    padded_train_labels = pad_sequences(
        train_encoded_labels, [0] * len(label_vocab), MAX_LENGTH
    )

    padded_test_sentences = pad_sequences(test_encoded_sentences, 0, MAX_LENGTH)
    padded_test_labels = pad_sequences(
        test_encoded_labels, [0] * len(label_vocab), MAX_LENGTH
    )

    if debug:
        # Print example data
        example_idx = 1  # Change this to see different examples

        print(
            f"Train data length: {len(train_sentences)}, Test data length: {len(test_sentences)}"
        )
        print(f"Example sentence: {' '.join(train_sentences[example_idx])}")
        print(f"Example labels: {train_raw_labels[example_idx]}")

        print(f"Encoded sentence: {train_encoded_sentences[example_idx]}")
        print(f"Encoded labels: {train_encoded_labels[example_idx]}")

        print(f"Padded sentence: {padded_train_sentences[example_idx]}")
        print(f"Padded labels: {padded_train_labels[example_idx]}")

    # Split training data into training and validation sets
    train_sentences, train_labels, val_sentences, val_labels = split_data(
        padded_train_sentences, padded_train_labels
    )

    if debug:
        print(
            f"Train data length: {len(train_sentences)}, Validation data length: {len(val_sentences)}"
        )

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
        padded_test_sentences,
        padded_test_labels,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        MAX_LENGTH,
        word_to_idx,
        idx_to_word,
        label_to_idx,
        idx_to_label,
    )


if __name__ == "__main__":
    prepare_data_pipeline(TRAIN_DATA_PATH, TEST_DATA_PATH, debug=True)
