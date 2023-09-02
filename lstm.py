from data import load_data, PAD
from metric import f1_score
import torch.nn as nn
import torch
import wandb


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
    model, train_loader, test_loader, optimizer, loss_function, num_epochs=5
):
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for sentences, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = torch.tensor(labels).view(-1)

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
        for sentences, labels in test_loader:
            with torch.no_grad():
                outputs = model(sentences)
            predictions = torch.argmax(outputs, dim=-1).tolist()
            all_predictions.extend(predictions)
            all_true_labels.extend(labels)

        validation_f1 = f1_score(all_predictions, all_true_labels)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training loss: {average_train_loss:.4f}")
        print(f"Validation F1 Score: {validation_f1:.4f}\n")


def main():
    # Initialize wandb
    wandb.init(
        project="DL-NLP-Final-Project-NER-Neural-Baseline",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "embedding_dim": 128,
            "hidden_dim": 256,
        },
    )
    config = wandb.config

    # Load data
    train_loader, test_loader, word_to_id, label_to_id = load_data()

    # Initialize the model
    model = LSTM_Model(len(word_to_id), len(label_to_id))
    print(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss(
        ignore_index=label_to_id[PAD]
    )  # Ignore the padding token for loss computation
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model for 5 epochs as a baseline
    train_model(
        model, train_loader, test_loader, optimizer, loss_function, num_epochs=5
    )


if __name__ == "__main__":
    main()
