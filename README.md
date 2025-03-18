## 🚀 Deployed App

The app is available on Streamlit (wake the app up if not inactive): [Click here to access](https://yodagpt.streamlit.app)

# YodaGPT
![alt text](https://github.com/danis-b/YodaGPT/blob/main/logo.png)

Welcome to YodaGPT, a text generation project that mimics the wise and  speech patterns of Jedi Master Yoda from Star Wars. Built using a transformer-based language model inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), this project trains on Yoda’s dialogue and generates text in his distinctive style via a Streamlit web interface.
This repository contains the training script, a pre-trained model, and a user-friendly app to generate Yoda-like wisdom from any prompt. Optimized with Optuna on a GPU cluster, YodaGPT achieves a validation loss ~1.2690 (val_loss - train_loss = 0.232), delivering coherent and Yoda-esque outputs.

Model Architecture:
* A decoder-only transformer with 3 layers of  multi-head self-attention (n_head=4) and feed-forward networks. 
* Embedding layer: Maps characters to n_embd=96 dimensions.
* Positional embedding: Encodes sequence position up to block_size=80.

Training:
* Dataset: quotes.txt split 90% train, 10% validation.
* Optimization: Optuna tuned parameters over 150 trials on a GPU cluster
  * learning_rate: 0.002259
  * n_embd: 96 (0.355 M parameters)
  * dropout: 0.2165
  * weight_decay: 0.05705
  * warmup_steps: 500
  * top_k: 2
  * block_size: 80
  * temperature: 0.89879
    
Generation:
* Top-K Sampling: Limits predictions to the top 2 probabilities (top_k=2)
* Temperature: Scales logits by ~0.89879, balancing coherence and creativity.
* Streamlit: Yields text character-by-character, displaying full words for a dynamic effect:
```
Fear is the path to the dark side.
To not make be is the dark side you not.
Truth be that is that it cannot.
The dark side is it you must be. Master the Force.
```
