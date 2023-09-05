import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from data_new import (
    prepare_data_pipeline,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PAD,
    tensor_to_sentences,
    tensor_to_labels,
)
import math


class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(SimpleRNNModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Simple RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Activation layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # Convert token indices to embeddings
        embedded = self.embedding(text)

        # Pass embeddings through RNN
        rnn_output, _ = self.rnn(embedded)

        # Pass RNN output through dense layer
        predictions = self.fc(rnn_output)

        # Sigmoid activation
        predictions = self.sigmoid(predictions)

        return predictions


def train(model, iterator, optimizer, criterion, device, idx_to_word, idx_to_label):
    """
    Training logic for an epoch
    """
    model.train()

    epoch_loss = 0

    for batch in iterator:
        sentences = batch["sentence"]
        labels = batch["label"]
        sentences, labels = sentences.to(device), labels.to(device)
        print(sentences.shape)
        print(tensor_to_sentences(sentences, idx_to_word))

        print(labels.shape)
        print(tensor_to_labels(labels, idx_to_label))

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(sentences)
        print(predictions.shape)
        exit()

        # Compute loss
        loss = criterion(
            predictions.view(-1, predictions.shape[-1]),
            labels.view(-1, labels.shape[-1]),
        )

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluation logic with micro-F1 score
    """
    model.eval()

    epoch_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            sentences = batch["sentence"]
            labels = batch["label"]
            sentences, labels = sentences.to(device), labels.to(device)

            predictions = model(sentences)

            # Convert sigmoid outputs to binary labels
            binary_predictions = (torch.sigmoid(predictions) >= 0.5).float()

            all_predictions.append(binary_predictions.view(-1).cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())

            loss = criterion(
                predictions.view(-1, predictions.shape[-1]),
                labels.view(-1, labels.shape[-1]),
            )
            epoch_loss += loss.item()

    # Compute micro-F1 score
    micro_f1 = f1_score(
        np.hstack(all_labels), np.hstack(all_predictions), average="micro"
    )

    return epoch_loss / len(iterator), micro_f1


def get_args():
    parser = argparse.ArgumentParser(description="Train an RNN baseline for NER.")
    parser.add_argument(
        "--train_path", type=str, required=False, help="Path to the training dataset."
    )
    parser.add_argument(
        "--test_path", type=str, required=False, help="Path to the test dataset."
    )
    return parser.parse_args()


def main_rnn(train_file_path, test_file_path):
    args = get_args()
    if args.train_path:
        train_file_path = args.train_path
    if args.test_path:
        test_file_path = args.test_path
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Call the prepare_data_pipeline to get data ready
    (
        train_loader,
        val_loader,
        test_loader,
        MAX_LENGTH,
        word_to_idx,
        idx_to_word,
        label_to_idx,
        idx_to_label,
    ) = prepare_data_pipeline(train_file_path, test_file_path)

    # 2. Define the RNN model and related components
    config = {}
    VOCAB_SIZE = len(word_to_idx)
    config["embedding_dim"] = 100
    config["hidden_dim"] = 128
    config["epochs"] = 100
    OUTPUT_DIM = len(label_to_idx)  # Number of labels
    PAD_IDX = word_to_idx[PAD]

    model = SimpleRNNModel(
        VOCAB_SIZE, config["embedding_dim"], config["hidden_dim"], OUTPUT_DIM, PAD_IDX
    )
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # 3. Execute the training loop
    for epoch in range(config["epochs"]):
        train_loss = train(
            model, train_loader, optimizer, criterion, device, idx_to_word, idx_to_label
        )
        val_loss, micro_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch: {epoch+1:02}")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}")
        print(f"\t Micro-F1 Score (Val): {micro_f1:.3f}")

    # 4. Evaluate on the test set
    test_loss, test_micro_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}")
    print(f"Micro-F1 Score (Test): {test_micro_f1:.3f}")

    return model


if __name__ == "__main__":
    main_rnn(TRAIN_DATA_PATH, TEST_DATA_PATH)
