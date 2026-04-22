#### FoLQA

The implementation of our method is in `evaluation/`, the dataset is under `dataset/`, and the dataset construction scripts are in `create_dataset/`.

#### FoLQA Baseline

The links to the official baseline code are as follows:

CoK: https://github.com/DAMO-NLP-SG/chain-of-knowledge

ToG: https://github.com/IDEA-FinAI/ToG

DoG: https://github.com/reml-group/DoG

#### Dataset
We use the FoLQA and CWQ datasets for our experiments. For experiments on the CWQ dataset, we randomly sampled 1,000 instances.


#### Package Description

This repository is structured as follows：

```text
FoLQA/
├─ create_dataset/                    Dataset construction pipeline
│  ├─ create_logic_query/             Generate logical queries for question construction
│  ├─ transform_data/                 Convert/normalize the format of logical queries
│  └─ create_question/                Build questions from logical queries and produce the dataset
├─ dataset/                            Data used for experiments
├─ KG/                                 Knowledge graph and related resources
├─ train_classifier/                   Train the Conjunction/Disjunction Detector
├─ train_classifier_negation/          Train the Negation Detector
└─ evaluation/                         Implementation of our method and evaluation scripts
```





