#### Environment

You can create the required environment for FOLR using the `environment.yml` file.

#### Preparation

**(1)LLM via Ollama**  

Use **Ollama** to serve **`llama3:8b`**. Make sure Ollama is installed, the model is pulled, and the server is running (default `http://localhost:11434`).  

Command: `ollama pull llama3:8b`

**(2)Detectors to prepare**

1) **Conjunction/Disjunction Detector** — used for question simplification in `simple_question/`
   - Base encoder: **BERT base (uncased)**: https://huggingface.co/google-bert/bert-base-uncased
   - Train with the scripts in `train_classifier/` and note the checkpoint path.

2) **Negation Detector**
   - Base encoder: **T5-base**: https://huggingface.co/google-t5/t5-base
   - Train with the scripts in `train_classifier_negation/` and note the checkpoint path.

**(3)Configure paths**

In the evaluation config (or CLI flags), set:

- Ollama endpoint (e.g., `http://localhost:11434`)
- LLM name: `llama3:8b`
- Paths to the trained detector checkpoints
- Dataset location under `dataset/`

**(4)Run evaluation**  

After setup, run `evaluation/eval.py`

#### How to run


After training both detectors, perform question simplification and configure the relevant paths in `evaluation/eval.py`; then run `evaluation/eval.py` to complete the evaluation.




