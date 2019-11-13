# Leveraging Model Interpretability and Stability to increase Model Robustness

This is the backbone code of the paper [Leveraging Model Interpretability and Stability to increase Model Robustness](https://arxiv.org/abs/1910.00387).
## Objective
- The purpose of this work is to detect potential prediction errors of a CNN and cancel those predictions.
- Inputs whose prediction are cancelled are not further processed by the CNN. If the area of application allows it (medical diagnosis, malware detection, autonomous driving), those inputs can be processed by a supplementary system (human specialist, radar/lidar).
## Overview
- The Conductance and the Label Change Rate (LCR) of a prediction are metrics used to determine if the prediction is likely to be wrong or correct
- [Conductance](https://arxiv.org/abs/1805.12233) rely on model interpretability and check if the prediction result was obtained by emphasizing corresponding model feature maps.
- [Label Change Rate](https://arxiv.org/abs/1812.05793) rely on the stability of the prediction to minor model parameter changes to determine if the prediction is wrong or not.
- After a CNN made a prediction on an input image, both LCR and conductance can be extracted from the process to examine whether the prediction is wrong or not.
![steps_colored](https://user-images.githubusercontent.com/34350063/68809582-853fb180-066c-11ea-8ae1-367ee9311645.png=200x200)
## Experimental results
- Performance of LCR, conductance, both LCR & conductance to distinguish wrong and correct prediction are displayed in the following graphs

## Imagenet file how to use
- Use 'training_conductance.py' and 'val_test_conductance.py' to calculate the conductance on image training, validation and test sets.
- To incorporate LCR, first use build_mutations.py to create modified version of the CNN.
- Then use 'mutant_prediction.py' to calculate the LCR of desired image set.
- After computing conductance (with or not LCR) please follow code in '100_class_imagenet.ipynb' (or 50) notebook to see how the binary classifier is trained and evaluated.

## cifar10 file how to use
- For now please ignore the scripts folder and use the 'conductance&mutation.ipynb' notebook to calculate conductance and LCR

## work in progress to make the repository's code easier to read
