def f1_score(predictions, true_labels, pad_label_id=0):
    """
    Manually compute the F1 score for sequence labeling tasks.

    Args:
    - predictions (list of lists): Predicted label IDs for each sentence.
    - true_labels (list of lists): True label IDs for each sentence.
    - pad_label_id (int): Label ID for padding tokens.

    Returns:
    - float: F1 score.
    """
    # Flatten the lists and remove padding label IDs
    flat_predictions = [
        label for sublist in predictions for label in sublist if label != pad_label_id
    ]
    flat_true_labels = [
        label for sublist in true_labels for label in sublist if label != pad_label_id
    ]

    # Compute precision, recall, and F1 score
    tp = sum(1 for p, t in zip(flat_predictions, flat_true_labels) if p == t)
    fp = sum(1 for p, t in zip(flat_predictions, flat_true_labels) if p != t)
    fn = sum(1 for p, t in zip(flat_predictions, flat_true_labels) if p != t)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return f1
