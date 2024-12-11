# LayerSkip
<a href='https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![License: CC BY-NC](https://img.shields.io/badge/License-CC_BY--NC-lightgrey.svg)](./LICENSE) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=oPxdfVVmLP8) [![arXiv](https://img.shields.io/badge/arXiv-2404.16710-b31b1b.svg)](https://arxiv.org/abs/2404.16710) [![alphaXiv](https://img.shields.io/badge/alphaXiv-2404.16710-9a2037.svg)](https://www.alphaxiv.org/abs/2404.16710)

This code base is the implementation of [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710).

<div align="center">
  <img src="https://github.com/user-attachments/assets/1fdd91d9-37ea-4b42-b5be-579fb5e1f2f2" width="500">
</div>

## Getting Started

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
