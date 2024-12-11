# Dynamic Early Exit and Layer Skipping

## Project Overview
This repository implements **Dynamic Early Exit** and **Dynamic Layer Skipping**, techniques designed to accelerate inference for **Large Language Models (LLMs)** like Llama2. These strategies enhance efficiency in real-time applications such as conversational AI, live translation, and code generation.

Building upon prior works like **LayerSkip**, we present a unified framework for dynamic inference optimization, achieving significant speedups while maintaining high-quality outputs across diverse tasks.

---

## Key Features

- **Dynamic Early Exit**:
  - Implements token-level confidence thresholds to dynamically exit inference early, optimizing resource usage for simpler inputs.
  
- **Dynamic Layer Skipping**:
  - Skips non-critical layers during inference, determined by input complexity and optimized through **Bayesian Optimization**.

- **Self-Speculative Decoding**:
  - Combines draft and verify mechanisms to accelerate decoding with minimal accuracy loss.

- **Interpretability Tools**:
  - Includes layer-wise token prediction visualizations and cosine similarity analysis for model behavior insights.

---
## Getting Started
   git clone https://github.com/Garyli2333/layer_skip.git
   cd layer_skip
   
- Setup environment:
```console
$ conda create --name layer_skip python=3.10
$ conda activate layer_skip

$ pip install -r requirements.txt
```

- Access models:
In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe. We provide 6 checkpoints on [HuggingFace](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a) of different Llama models continually pretrained using the LayerSkip recipe:

    - [`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
    - [`facebook/layerskip-llama2-13B`](https://huggingface.co/facebook/layerskip-llama2-13B)
    - [`facebook/layerskip-codellama-7B`](https://huggingface.co/facebook/layerskip-codellama-7B)
    - [`facebook/layerskip-codellama-34B`](https://huggingface.co/facebook/layerskip-codellama-34B)
    - [`facebook/layerskip-llama3-8B`](https://huggingface.co/facebook/layerskip-llama3-8B)
    - [`facebook/layerskip-llama3.2-1B`](https://huggingface.co/facebook/layerskip-llama3.2-1B)

In order to access each model:

1. Visit the model's corresponding link above, make sure you are logged on the HuggingFace website with your account.
2. Fill the request form and submit it. Approval may take a while and you should receive an email notification to notify you that permission to the model is granted.
3. Follow the steps [here](https://huggingface.co/docs/hub/en/security-tokens) to obtain a user access token.
4. In the command-line run `huggingface-cli login`, and you will be prompted to provide the token you have obtained in Step 3.

Once you run those steps, the commands below to run the LayerSkip checkpoints should work.

```
