
## Preprocessing Functions

1. **`load_data`**: 
    - **Input**: File paths for training and testing JSON files.
    - **Output**: Loaded JSON data for training and testing.
    - **Purpose**: Load the JSON files into Python objects for further processing.
  
2. **`extract_sentences_and_labels`**: 
    - **Input**: Loaded JSON data for training and testing.
    - **Output**: Lists of sentences and corresponding raw label data for training and testing.
    - **Purpose**: Extract sentences and raw labels from the loaded JSON objects.

3. **`generate_label_vocab`**: 
    - **Input**: Raw label data from training and testing.
    - **Output**: A vocabulary of unique labels found in the dataset.
    - **Purpose**: Generate a list of unique labels for encoding.

4. **`encode_labels`**: 
    - **Input**: Raw label data and label vocabulary.
    - **Output**: Multi-hot encoded labels for each token in the sentences.
    - **Purpose**: Convert the raw label data to a format that can be used by the neural network.

5. **`pad_sequences`**: 
    - **Input**: List of tokenized sentences and their corresponding multi-hot encoded labels.
    - **Output**: Padded sentences and labels.
    - **Purpose**: Pad the tokenized sentences and labels to a fixed length for batch processing.

6. **`find_max_length`**:
    - **Input**: List of tokenized sentences.
    - **Output**: Maximum length among all sentences.
    - **Purpose**: Find the maximum length among all sentences for dynamic padding.

## Utility Functions

1. **`sample_data`**:
    - **Input**: Sentences and labels.
    - **Output**: Randomly sampled subset of sentences and labels.
    - **Purpose**: Easily inspect a subset of the data for quick debugging.

2. **`tensor_to_sentences`**:
    - **Input**: Mini-batched data tensor representing tokenized sentences.
    - **Output**: List of natural language sentences.
    - **Purpose**: Transform mini-batched data tensor into readable sentences for debugging or analysis.

3. **`tensor_to_labels`**:
    - **Input**: Mini-batched data tensor representing multi-hot encoded labels.
    - **Output**: List of label annotations in natural language.
    - **Purpose**: Transform mini-batched data tensor into readable labels for debugging or analysis.

## Main Function

- **Input**: None
- **Output**: Preprocessed and padded sentences and labels for both training and testing.
- **Purpose**: To showcase the complete preprocessing pipeline.

7. **`encode_sentences`**:
    - **Input**: List of tokenized sentences and a vocabulary of unique words.
    - **Output**: Encoded sentences where each word is replaced by its index in the vocabulary.
    - **Purpose**: Convert the tokenized sentences into a format that can be fed into the neural network.

8. **`build_word_to_idx`**:
    - **Input**: List of tokenized sentences.
    - **Output**: A dictionary mapping each unique word to a unique index.
    - **Purpose**: Build a mapping from words to indices based on the dataset.
    
9. **`build_idx_to_word`**:
    - **Input**: `word_to_idx` dictionary.
    - **Output**: A dictionary mapping each index back to its corresponding word.
    - **Purpose**: Build a reverse mapping from indices to words for decoding.
    
10. **`build_label_to_idx`**:
    - **Input**: List of unique labels (label vocabulary).
    - **Output**: A dictionary mapping each label to a unique index.
    - **Purpose**: Build a mapping from labels to indices.
    
11. **`build_idx_to_label`**:
    - **Input**: `label_to_idx` dictionary.
    - **Output**: A dictionary mapping each index back to its corresponding label.
    - **Purpose**: Build a reverse mapping from indices to labels for decoding.
