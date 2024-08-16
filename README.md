# Makemore
Building Karpathy's Makemore -  20th Summer Project by Han 2024

This is my note on learning Makemore and implementing it step by step

Makemore originally created by Karpathy, et al.

Written in Indonesian+English

## What I learned
- Torch syntax (I found out some is similar to Numpy), indexing, matrix (tensor) multiply, and documentation, also process on interpreting tensor size
- Bigram language model with creating dict of char and look up table
- Sampling using torch multinomial
- One hot encoding
- How torch generator seed works
- Average Negative log likelihood (dari log likelihood sampe ketemu avg)
- Model smoothing Bigram
- Making simple linear bigram neural network and gradient descent using Torch
- Softmax activation Function
- Building 3 word and 1 label (next word) dataset
- Embedding matrix for the characters
- Building neural network from scratch using torch.tensor
- Reducing dimension of tensor from example, [228146, 3, 2] to [228146, 6] via view() method
- Average Negative log likelihood + cross entropy loss
- Cross entropy function is better in numerical stability:
    Sebenernya karena ini probabilitas outputnya (udah normalized) jadi kita kalau modif modif angkanya akan sama
    Kita bisa coba modif angka angka di logits kasih +10 dan lihat hasilnya akan probabilitas yang sama
    Jadi torch bakal otomatis kalkulasi nilai max yang ada di logits itu dan kurangin logits berdasarkan max itu
- Splitting dataset (Train, val, test)
- Methods to make calculation in GPU
- Finding good learning rate
- Training using mini batch approach
- Vanishing gradient problem, analyzing the weakness of Tanh which is squashing function and how to fix it with He/kaiming Init or adding batch normalization layer
- Little bit of dying relu and dead neuron
- Batch Normalization Layer in depth and 2 approach on estimating mean and standard deviation for inference on the batchnorm layer
---
Noted and Created by Han Summer 2024

Part of The 20th Summer Project