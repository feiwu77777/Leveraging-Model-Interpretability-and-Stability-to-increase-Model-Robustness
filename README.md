# Leveraging Model Interpretability and Stability to increase Model Robustness

This is the backbone code of the paper [Leveraging Model Interpretability and Stability to increase Model Robustness](https://arxiv.org/abs/1910.00387).
## Objective
- The purpose of this work is to detect potential prediction errors of a CNN and cancel those predictions.
- Inputs whose prediction are cancelled are not further processed by the CNN. If the area of application allows it (medical diagnosis, malware detection, autonomous driving), those inputs can be processed by a supplementary system (human specialist, radar/lidar).
## Overview
- The Conductance and the Label Change Rate (LCR) of a prediction are metrics used to determine if the prediction is likely to be wrong or correct.
- [Conductance](https://arxiv.org/abs/1805.12233) rely on model interpretability and check if the prediction result was obtained by emphasizing corresponding model feature maps.
- [Label Change Rate](https://arxiv.org/abs/1812.05793) rely on the stability of the prediction to minor model parameter changes to determine if the prediction is wrong or not.
- After a CNN (IncepionV3) made a prediction on an input image, both LCR and Conductance can be extracted from the process to examine whether the prediction is wrong or not.
![steps_colored](https://user-images.githubusercontent.com/34350063/68809582-853fb180-066c-11ea-8ae1-367ee9311645.png)
## Experimental results
- Performance (measured by AUROC) of LCR, Conductance, both LCR & Conductance to distinguish wrong and correct predictions of 3 CNN trained over 3 datasets (CIFAR10, 50 classes ImageNet, 100 classes ImageNet are displayed in the following graphs:
![Screenshot from 2019-11-13 23-38-57](https://user-images.githubusercontent.com/34350063/68810632-f54f3700-066e-11ea-900a-02a6efc3cb60.png)
- Results of using LCR & Conductance to classify test set predictions of the 3 CNN (trained over the 3 respective datasets) are displayed in the following table:
![Screenshot from 2019-11-14 14-14-03](https://user-images.githubusercontent.com/34350063/68860117-21a79980-06e9-11ea-9ac2-9e90c84ce78d.png)

## cifar10 file how to use
- Follow the 'conductance&mutation.ipynb' notebook to see how conductance and LCR of data are calculated and used to train an error detector to differentiate wrong and correct predictions of a CNN
- To build modified CNNs (mutants) that will be used to calculate LCR run
```bash
python build_mutations_cifar.py
```
## Imagenet file how to use
- Use 'training_conductance.py' and 'val_test_conductance.py' to calculate the conductance on image training, validation and test sets.
- To incorporate LCR, first use build_mutations.py to create modified version of the CNN.
- Then use 'mutant_prediction.py' to calculate the LCR of desired image set.
- After computing conductance (with or not LCR) please follow code in '100_class_imagenet.ipynb' (or 50) notebook to see how the binary classifier is trained and evaluated.

## work in progress to make the repository's code easier to read.
