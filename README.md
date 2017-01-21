# adversarial-experiments
Experiments with adversarial models

## Rock Paper Scissors
Rock-Paper-Scissors is a multiplayer game that is not guaranteed to converge using gradient descent.
* `RockPaperScissors.m` plays rock-paper-scissors using softmax to constrain parameters
* `RockPaperScissorsBarrier.m` plays

[Video](https://youtu.be/JmON4S0kl04) of playing Rock-Paper-Scissors by gradient descent

## Simple Generative Adversarial Network (GAN)

`SimpleGAN.m` trains a GAN to generate a single 2D datapoint.

Changes to learning rates and regularization determine whether model converges, oscillates, or trains slowly.

[Video](https://youtu.be/ebMei6bYeWw) of simple GAN training
