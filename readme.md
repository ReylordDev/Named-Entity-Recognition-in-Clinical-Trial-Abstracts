# Deep Learning Project: Named Entity Recognition in Clinical Trial Abstracts

This project first evaluates the performance of NER in Clinical Trial Abstracts using a simple neural baseline and a pretrained BERT baseline model. Then the goal was to improve the overall performance using a creative approach. The chosen approach here involves creating additional and artificial training data using GPT-4 and training a BERT model on the full data. 

For more details and results comparison, check the [final paper](dl_nlp_ner_paper_lklocke.pdf).

## Task Description

This task is about extracting named entities from clinical trial abstracts. Here an entity is described by a token sequence within the corresponding sentence and a label. 

The extraction results are evaluated by the micro F1 measure.

The format of the train and test data json file is as follows:

```

- the top level object is an array of abstract objects

- abtract objects represent annotated clinical trial abstracts and have the following fields:
  -- abstract_id: id of clinical trial abstract
  -- sentences: array of sentence objects
  
- sentence objects represent sentences of an abstract and have the following fields:
  -- sentence_id: id of sentence (only uniuqe within abstract)
  -- words: array of strings
  -- entities: array of entity objects; entities of the sentence
  
- entity objects represent named entities and have the following fields:
  -- start_pos/end_pos: word offsets within sentence of start/end position of entity (both inclusive)
  -- label: label(class) of entity; 
  -- words: array of strings representing named entity

```
