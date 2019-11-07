# Leveraging Model Interpretability and Stability to increase Model Robustness

This is the backbone code of the paper [Leveraging Model Interpretability and Stability to increase Model Robustness](https://arxiv.org/abs/1910.00387).
## Objective
- The purpose of this work is to detect potential prediction errors of a CNN and cancel those predictions.
- Inputs whose prediction are cancelled are not further processed by the CNN. Depending on the application, those inputs can be processed by a human.
## How to detect wrong prediction
![matchstick](match_cond.png)
![matchstick](orange_cond.png)
- Above images of a matchstick and oranges are both predicted as a matchstick object.
- Through a serie of experiments, feature maps 720, 1232 and 1257 are identified as feature maps that react strongly when a matchstick image is give to the CNN.
- We can observe that the prediction of the orange image doesn't trigger those feature maps meaning that the prediction of this image is likely false.


## Imagenet file how to use
- First, the image dataset divided into training, validation and test set and the CNN is trained with the image training set.
- Then use 'training_conductance.py' and 'val_test_conductance.py' to calculate the conductance on all 3 sets.
- To incorporate LCR, first use build_mutations.py to create modified version of the CNN.
- Then use 'mutant_prediction.py' to calculate the LCR of desired image set.
- After computing conductance (with or not LCR) please follow code in '100_class_imagenet.ipynb' (or 50) notebook to see how the binary classifier is trained and evaluated.

## cifar10 file how to use
- For now please ignore the scripts folder and use the 'conductance&mutation.ipynb' notebook to calculate conductance and LCR

## work in progress to make the repository's code easier to read
