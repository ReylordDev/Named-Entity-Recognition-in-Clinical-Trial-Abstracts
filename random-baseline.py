from data import load_data
from sklearn.metrics import f1_score
import wandb
import argparse
import numpy as np


class RandomProbabilisticClassifier:
    def __init__(self, num_classes):
        """Initialize the classifier with the number of classes."""
        self.num_classes = num_classes

    def predict_proba(self, X):
        """Predict random probability distributions for each integer in each sample in X."""
        batch_size, seq_length = X.shape
        random_probs = np.random.rand(batch_size, seq_length, self.num_classes)
        return random_probs / random_probs.sum(axis=2, keepdims=True)

    def predict(self, X):
        """Predict class labels based on the highest probability for each integer in each sample."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=2)


def main(args=None):
    # Initialize wandb
    wandb.init(
        project="DL-NLP-Neural-Baseline",
        name="random-baseline-4",
    )

    # Load data
    if args and args.train_path and args.test_path:
        train_loader, test_loader, word_to_id, label_to_id, _ = load_data(
            args.train_path, args.test_path
        )
    else:
        train_loader, test_loader, word_to_id, label_to_id, _ = load_data()

    num_classes = len(label_to_id)

    # Initialize the random classifier
    clf = RandomProbabilisticClassifier(num_classes=num_classes)

    # Predict and evaluate
    all_true_labels = []
    all_pred_labels = []

    for sentences, true_labels in test_loader:
        pred_labels = clf.predict(sentences)
        all_true_labels.extend(true_labels.numpy())
        all_pred_labels.extend(pred_labels)

    flat_true_labels = np.array(all_true_labels).flatten()
    flat_pred_labels = np.array(all_pred_labels).flatten()

    non_pad_indices = flat_true_labels != label_to_id["PAD"]
    flat_true_labels = flat_true_labels[non_pad_indices]
    flat_pred_labels = flat_pred_labels[non_pad_indices]

    # Evaluate
    print(
        f"F1 score: {f1_score(flat_true_labels, flat_pred_labels, average='micro'):.4f}"
    )
    wandb.log(
        {"test_f1": f1_score(flat_true_labels, flat_pred_labels, average="micro")}
    )


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
