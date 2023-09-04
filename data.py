import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

TRAIN_DATA_PATH = "data/train.json"
TEST_DATA_PATH = "data/test.json"
PAD = "<PAD>"


class ClinicalTrialsDataset(Dataset):
    def __init__(self, sentences, labels):
        """
        Constructor method to initialize the dataset with sentences and labels.

        Args:
        - sentences (list of lists): List of tokenized sentences, where each sentence is a list of words.
        - labels (list of lists): List of labels associated with each sentence.
        """
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Given an index idx, returns the sentence and its corresponding label at that index.

        Args:
        - idx (int): Index of the desired sample.

        Returns:
        - tuple: A tuple containing the sentence and its associated label at the given index.
        """
        return self.sentences[idx], self.labels[idx]


def load_data(
    train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH, batch_size=32
):
    # Load the datasets from the provided JSON files
    with open(train_data_path, "r") as train_file:
        train_dataset = json.load(train_file)

    with open(test_data_path, "r") as test_file:
        test_dataset = json.load(test_file)

    (
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
        word_to_id,
        label_to_id,
        max_sequence_length,
    ) = preprocess_data(train_dataset, test_dataset)

    train_loader, test_loader = get_dataloaders(
        train_sentences, train_labels, test_sentences, test_labels, batch_size
    )

    print("Data loaded.")

    return train_loader, test_loader, word_to_id, label_to_id, max_sequence_length


def preprocess_data(train_dataset, test_dataset):
    (
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
    ) = extract_sentences_and_labels(train_dataset, test_dataset)

    word_to_id, label_to_id = create_vocabulary(
        train_sentences, train_labels, test_sentences, test_labels
    )

    (
        train_sentences_padded,
        train_labels_int,
        test_sentences_padded,
        test_labels_int,
        max_sequence_length,
    ) = convert_and_pad_data(
        train_sentences,
        train_labels,
        test_sentences,
        test_labels,
        word_to_id,
        label_to_id,
    )

    return (
        train_sentences_padded,
        train_labels_int,
        test_sentences_padded,
        test_labels_int,
        word_to_id,
        label_to_id,
        max_sequence_length,
    )


def extract_sentences_and_labels(train_dataset, test_dataset):
    # Extract sentences and labels from the datasets
    train_sentences = []
    train_labels = []
    for abstract in train_dataset:
        for sentence in abstract["sentences"]:
            train_sentences.append(sentence["words"])
            train_labels.append([entity["label"] for entity in sentence["entities"]])

    test_sentences = []
    test_labels = []
    for abstract in test_dataset:
        for sentence in abstract["sentences"]:
            test_sentences.append(sentence["words"])
            test_labels.append([entity["label"] for entity in sentence["entities"]])

    print("Data extracted.")
    print("Train sentence count: ", len(train_sentences))
    print("Test sentence count: ", len(test_sentences))

    return train_sentences, train_labels, test_sentences, test_labels


def create_vocabulary(train_sentences, train_labels, test_sentences, test_labels):
    # Create a vocabulary of unique words from train and test datasets
    word_to_id = defaultdict(lambda: len(word_to_id))
    word_to_id[PAD]  # Initialize with padding token

    # Populate the vocabulary
    for sentence in train_sentences + test_sentences:
        for word in sentence:
            word_to_id[word]

    # Create a set of unique labels to convert labels into integers
    label_to_id = defaultdict(lambda: len(label_to_id))
    for label_list in train_labels + test_labels:
        for label in label_list:
            label_to_id[label]

    print("Vocabulary and label set created.")
    print("Vocabulary size: ", len(word_to_id))
    print("Label set size: ", len(label_to_id) + 1)  # +1 for the padding token

    return word_to_id, label_to_id


def convert_and_pad_data(
    train_sentences, train_labels, test_sentences, test_labels, word_to_id, label_to_id
):
    # Convert sentences to their corresponding integer values
    train_sentences_int = [
        [word_to_id[word] for word in sentence] for sentence in train_sentences
    ]
    test_sentences_int = [
        [word_to_id[word] for word in sentence] for sentence in test_sentences
    ]

    # Convert labels to their corresponding integer values
    train_labels_padded = [
        [label_to_id[label] for label in label_list] for label_list in train_labels
    ]
    test_labels_padded = [
        [label_to_id[label] for label in label_list] for label_list in test_labels
    ]

    # Pad sequences to have the same length
    max_sequence_length = max(
        max(len(sentence) for sentence in train_sentences_int),
        max(len(sentence) for sentence in test_sentences_int),
    )
    train_sentences_padded = [
        sentence + [word_to_id[PAD]] * (max_sequence_length - len(sentence))
        for sentence in train_sentences_int
    ]
    test_sentences_padded = [
        sentence + [word_to_id[PAD]] * (max_sequence_length - len(sentence))
        for sentence in test_sentences_int
    ]

    # Pad the labels to the same max sequence length as the sentences
    train_labels_padded = [
        label_list + [label_to_id[PAD]] * (max_sequence_length - len(label_list))
        for label_list in train_labels_padded
    ]
    test_labels_padded = [
        label_list + [label_to_id[PAD]] * (max_sequence_length - len(label_list))
        for label_list in test_labels_padded
    ]

    print("Data converted and padded.")
    print("Max sequence length: ", max_sequence_length)

    return (
        train_sentences_padded,
        train_labels_padded,
        test_sentences_padded,
        test_labels_padded,
        max_sequence_length,
    )


def get_dataloaders(
    train_sentences_padded,
    train_labels_padded,
    test_sentences_padded,
    test_labels_padded,
    batch_size,
):
    # Convert data into PyTorch tensors and set up dataloaders
    train_sentences_tensor = torch.tensor(train_sentences_padded)
    test_sentences_tensor = torch.tensor(test_sentences_padded)
    train_labels_tensor = torch.tensor(train_labels_padded)
    test_labels_tensor = torch.tensor(test_labels_padded)
    train_dataset = ClinicalTrialsDataset(train_sentences_tensor, train_labels_tensor)
    test_dataset = ClinicalTrialsDataset(test_sentences_tensor, test_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Dataloaders set up.")
    print("Train dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))

    return train_loader, test_loader


def ids_to_sentences(ids, id_to_word):
    """
    Given a tensor of shape [seq_length], returns the corresponding sentence as a string.

    Args:
    - ids (torch.Tensor): Tensor of shape [seq_length] containing the word ids of the sentence.
    - id_to_word (dict): Dictionary mapping word ids to words.

    Returns:
    - str: The sentence corresponding to the given word ids.
    """
    return " ".join([id_to_word[id] for id in ids])


def ids_to_labels(ids, id_to_label):
    """
    Given a tensor of shape [seq_length], returns the corresponding labels as a string.

    Args:
    - ids (torch.Tensor): Tensor of shape [seq_length] containing the label ids of the sentence.
    - id_to_label (dict): Dictionary mapping label ids to labels.

    Returns:
    - str: The labels corresponding to the given label ids.
    """
    return " ".join([id_to_label[id] for id in ids])


if __name__ == "__main__":
    load_data()
