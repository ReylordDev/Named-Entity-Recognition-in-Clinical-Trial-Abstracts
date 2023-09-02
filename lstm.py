from data import load_data, PAD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import wandb
import argparse


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim=128, hidden_dim=256):
        super(LSTM_Model, self).__init__()

        # Embedding layer to convert words to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Dense layer for classification
        self.fc = nn.Linear(hidden_dim, label_size)

    def forward(self, x):
        # Pass input through embedding layer
        x = self.embedding(x)

        # Pass embeddings through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass LSTM outputs through dense layer
        out = self.fc(lstm_out)

        return out


def train_model(
    model,
    train_loader,
    optimizer,
    loss_function,
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

    wandb.watch(model, loss_function, log="all", log_freq=10)

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[-1])
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

        # Flatten lists for metric computation
        flat_predictions = [label for sublist in all_predictions for label in sublist]
        flat_true_labels = [label for sublist in all_true_labels for label in sublist]

        validation_f1 = f1_score(flat_predictions, flat_true_labels, average="micro")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training loss: {average_train_loss:.4f}")
        print(f"Validation F1 Score: {validation_f1:.4f}\n")
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": average_train_loss,
                "val_f1": validation_f1,
            }
        )


def evaluate_model(model, test_loader, device):
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

    # Flatten lists for metric computation
    flat_predictions = [label for sublist in all_predictions for label in sublist]
    flat_true_labels = [label for sublist in all_true_labels for label in sublist]

    f1_micro = f1_score(flat_true_labels, flat_predictions, average="micro")

    wandb.log({"test_f1": f1_micro})


def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="DL-NLP-LSTM",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "epochs": 100,
        },
    )
    config = wandb.config

    # Load data
    if args:
        train_loader, test_loader, word_to_id, label_to_id = load_data(
            args.train_path, args.test_path
        )
    else:
        train_loader, test_loader, word_to_id, label_to_id = load_data()

    # Initialize the model
    model = LSTM_Model(len(word_to_id), len(label_to_id))
    model.to(device)
    print(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss(
        ignore_index=label_to_id[PAD]
    )  # Ignore the padding token for loss computation
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model for config.epochs as a baseline
    train_model(
        model,
        train_loader,
        optimizer,
        loss_function,
        device=device,
        num_epochs=config.epochs,
        val_split=0.2,
    )

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device=device)


def get_args():
    parser = argparse.ArgumentParser(description="Train an LSTM model for NER.")
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
