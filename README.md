# Leveraging Model Interpretability and Stability to increase Model Robustness

This is the backbone code of the paper [Leveraging Model Interpretability and Stability to increase Model Robustness](https://arxiv.org/abs/1910.00387).

## Imagenet file how to use
- First, the image dataset divided into training, validation and test set and the CNN is trained with the image training set.
- Then use 'training_conductance.py' and 'val_test_conductance.py' to calculate the conductance on all 3 sets.
- To incorporate LCR, first use build_mutations.py to create modified version of the CNN.
- Then use 'mutant_prediction.py' to calculate the LCR of desired image set.

## cifar10 file how to use
- For now please ignore the scripts folder and use the 'conductance&mutation.ipynb' notebook to calculate conductance and LCR

## work in progress to make the repository's code easier to read
