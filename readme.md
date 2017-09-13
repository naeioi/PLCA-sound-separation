**Semi-Supervised Separation of Sounds from Single-Channel Mixtures By Probability Latent Component Analysis**

-----

## Introduction

This an implementaion of *Probability Latent Component Analysis* model, which is essentially a EM-style non-negative matrix factorization algorithm. This example extracts piano sounds from single-channel mixed of music.

This is done by first learning P(f|z) from piano samples,
which is later used to partially fix P(f|z) in training on mixed music. 
Then reconstruct spectrogram from P(f|z), P(t|z) and P(z) where P(z') = 0 for all z' other than piano features.

## Usage
Dependency: numpy, scipy, matplotlib

You should install Jupyter notebook to run the code in `example.py`.

---------

Credit: Spectrogram construct By Frank Zalkow. Please refer to http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?i=1


[1] P. Smaragdis, B. Raj and M. Shashanka, A Probabilistic Latent Variable Model for Acoustic Modeling, Proc. Neural Information Processing Systems Workshop on Advances in Models for Acoustic Processing, 2006.