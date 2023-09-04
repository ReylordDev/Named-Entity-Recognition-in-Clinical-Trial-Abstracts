from data import load_data, PAD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
import wandb
import argparse


class FeedForwardNN_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        label_size,
        max_sequence_length,
        embedding_dim=128,
        hidden_dim=256,
    ):
        super(FeedForwardNN_Model, self).__init__()

        # Embedding layer to convert words to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # # Flattening the sequence
        # self.flatten = nn.Flatten()

        # Dense layers with ReLU activations
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")

        # You can add more layers if needed using the pattern above

        # Output layer for classification
        self.fc2 = nn.Linear(hidden_dim, label_size)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # Pass input through embedding layer
        x = self.embedding(x)

        # # Flatten the sequence
        # x = self.flatten(x)

        # Pass through dense layers with ReLU activation
        x = self.fc1(x)
        x = self.relu1(x)

        # Pass through output layer
        out = self.fc2(x)

        return out


def train_model(
    model,
    train_loader,
    optimizer,
    loss_function,
    word_to_id,
    label_to_id,
    device,
    num_epochs,
    val_split=0.2,
):
    # Overriding the train loader in a sketchy makeshift fix
    train_dataset = train_loader.dataset
    print(f"Train dataset size: {len(train_dataset)}")

    # Split the training dataset into training and validation
    train_data, val_data = train_test_split(train_dataset, test_size=val_split)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    id_to_word = {v: k for k, v in word_to_id.items()}
    id_to_label = {v: k for k, v in label_to_id.items()}

    # wandb.watch(model, loss_function, log="all", log_freq=10)

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)

            sentence_ints = sentences[0].cpu().numpy().tolist()
            sentence = " ".join([id_to_word[id] for id in sentence_ints])
            labels_ints = labels[0].cpu().numpy().tolist()
            labels_str = [id_to_label[id] for id in labels_ints]
            if epoch == num_epochs - 1:
                print(sentence_ints)
                print(sentence)
                print(labels_ints)
                print(labels_str)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentences)

            outputs = outputs.view(-1, outputs.shape[-1])
            if epoch == num_epochs - 1:
                outputs_probs = torch.softmax(outputs, dim=-1)
                most_likely_labels = torch.argmax(outputs_probs, dim=-1)
                output_labels_ints = most_likely_labels.cpu().detach().numpy().tolist()
                output_labels = " ".join([id_to_label[id] for id in output_labels_ints])
                print(output_labels_ints)
                print(output_labels)
                exit()

            labels = labels.clone().detach().to(device).view(-1)

            # Compute loss, gradients, and update parameters
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_predictions = []
        all_true_labels = []
        for sentences, labels in val_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(sentences)
            predictions = torch.argmax(outputs, dim=-1).cpu().numpy().tolist()
            true_labels = labels.cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

        flat_true_labels = np.array(all_predictions).flatten()
        flat_pred_labels = np.array(all_true_labels).flatten()

        non_pad_indices = flat_true_labels != label_to_id["PAD"]
        flat_true_labels = flat_true_labels[non_pad_indices]
        flat_pred_labels = flat_pred_labels[non_pad_indices]

        validation_f1 = f1_score(flat_true_labels, flat_pred_labels, average="micro")

        if (epoch + 1) % 20 == 0 or epoch + 1 == num_epochs:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training loss: {average_train_loss:.4f}")
            print(f"Validation F1 Score: {validation_f1:.4f}\n")
            # wandb.log(
            #     {
            #         "epoch": epoch + 1,
            #         "train_loss": average_train_loss,
            #         "val_f1": validation_f1,
            #     }
            # )


def evaluate_model(model, test_loader, label_to_id, device):
    """Evaluate the model on the test set."""

    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_true_labels = []

    for sentences, labels in test_loader:
        sentences, labels = sentences.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(sentences)

        predictions = torch.argmax(outputs, dim=-1).cpu().numpy().tolist()
        true_labels = labels.cpu().numpy().tolist()

        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

    flat_true_labels = np.array(all_predictions).flatten()
    flat_pred_labels = np.array(all_true_labels).flatten()

    non_pad_indices = flat_true_labels != label_to_id["PAD"]
    flat_true_labels = flat_true_labels[non_pad_indices]
    flat_pred_labels = flat_pred_labels[non_pad_indices]

    f1_micro = f1_score(flat_true_labels, flat_pred_labels, average="micro")

    # wandb.log({"test_f1": f1_micro})


def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    # wandb.init(
    #     project="DL-NLP-Neural-Baseline",
    #     name="feed-forward-baseline-4",
    #     config={
    #         "learning_rate": 0.00001,
    #         "batch_size": 32,
    #         "embedding_dim": 128,
    #         "hidden_dim": 256,
    #         "epochs": 500,
    #     },
    # )
    # config = wandb.config

    # Load data
    if args:
        (
            train_loader,
            test_loader,
            word_to_id,
            label_to_id,
            max_sequence_length,
        ) = load_data(args.train_path, args.test_path)
    else:
        (
            train_loader,
            test_loader,
            word_to_id,
            label_to_id,
            max_sequence_length,
        ) = load_data()

    # Initialize the model
    model = FeedForwardNN_Model(len(word_to_id), len(label_to_id), max_sequence_length)
    model.to(device)
    print(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss(
        ignore_index=label_to_id[PAD]
    )  # Ignore the padding token for loss computation
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model for config.epochs as a baseline
    train_model(
        model,
        train_loader,
        optimizer,
        loss_function,
        word_to_id,
        label_to_id,
        device=device,
        # num_epochs=config.epochs,
        num_epochs=500,
        val_split=0.2,
    )

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, label_to_id, device=device)


def get_args():
    parser = argparse.ArgumentParser(description="Train Feed-Forward model for NER.")
    parser.add_argument(
        "--train_path", type=str, required=False, help="Path to the training dataset."
    )
    parser.add_argument(
        "--test_path", type=str, required=False, help="Path to the test dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
