# Smol World Model  
![Gemini_Generated_Image_fv44b0fv44b0fv44](https://github.com/user-attachments/assets/a0d3fe8b-ad57-4923-908f-b74f308b30d7)

A lightweight implementation of a **World Model** for a Pacman-like game. Inspired by **Google Genie**, this project demonstrates how to build a generative interactive environment using a VQ-VAE for visual compression and a Transformer for state prediction.

## 🧠 Architecture

The architecture adapts concepts from state-of-the-art generative world models:

1.  **Environment (`game/`)**: A custom Pacman-like game simulation.
2.  **Data Collection (`datacollection/`)**: An agent explores the environment to collect a dataset of observations (frames) and actions.
3.  **Visual Encoder (`tokenizer/`)**: A **VQ-VAE** (Vector Quantized Variational Autoencoder) compresses the high-dimensional game frames into a sequence of discrete latent tokens.
4.  **Dynamics Model (`transformer/`)**: A **Transformer** learns to predict the next token (future frame) given the past tokens, effectively simulating the game logic within the neural network.

## 📂 Project Structure

```bash
Smol-World-Model/
├── datacollection/   # Scripts to run the game and save training data
├── game/             # The Pacman-like game environment implementation
├── tokenizer/        # VQ-VAE model definition and training scripts
└── transformer/      # Transformer model definition and training scripts
