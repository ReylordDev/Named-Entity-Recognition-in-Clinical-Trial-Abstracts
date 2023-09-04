## Data Format Description for Clinical Trial Abstracts NER

### Raw JSON Format
- **abstract_id**: Unique identifier for each clinical trial abstract.
- **sentences**: List of sentence objects.
  - **sentence_id**: Unique identifier for each sentence within an abstract.
  - **words**: List of tokens (words and symbols) in the sentence.
  - **entities**: List of label spans in the sentence.
    - **start_pos**: Start position of the labeled span in the sentence.
    - **end_pos**: End position of the labeled span in the sentence.
    - **label**: The label category.
    - **words**: Words in the labeled span.

### Processed Data
1. **Sentences**: Tokenized sentences, each represented as a list of words.
2. **Labels**: Associated labels for each token in the sentences, multi-hot encoded.

### Encodings
- **Words**: Mapped to integers based on a vocabulary index.
- **Labels**: Multi-hot encoded based on label vocabulary.

### Padding
- All sequences padded to a dynamic maximum length (`MAX_LENGTH`), determined by the longest sentence in the dataset.

### Special Tokens
- **PAD**: Used for padding shorter sequences. Mapped to integer 0.
- **`<UNK>`**: Represents unknown words during encoding.

### File Paths
- **TRAIN_DATA_PATH**: Path to the training data JSON file.
- **TEST_DATA_PATH**: Path to the test data JSON file.
