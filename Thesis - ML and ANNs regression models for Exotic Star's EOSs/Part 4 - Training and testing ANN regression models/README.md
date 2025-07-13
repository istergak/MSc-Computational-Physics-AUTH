# Part 4 - Training and testing deep learning regression models

In this directory we present all our codes for building and fitting deep learning regression models, using our regression data.
For more about how these data where produced, see [Part 1](https://github.com/istergak/MSc-Computational-Physics-AUTH/tree/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars) and [Part 2](https://github.com/istergak/MSc-Computational-Physics-AUTH/tree/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars).

**Notes**

Files of regression data must be included in the same directory with the following modules and notebooks, for the algorithms to run properly.

**Links to regression data**

See [here](https://drive.google.com/drive/folders/1eFYPW1juSy4aSwTBDs-ye0ToogRbSJfv)

## Modules

[data_analysis_ES_ANNs.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%204%20-%20Training%20and%20testing%20ANN%20regression%20models/data_analysis_ES_ANNs.py): module containing functions and classes for **a)** assessing linear correlations in regression data, **b)** training and testing deep learning regression models and **c)** storaging the fitting results in .pkl files and loading the .pkl files of trained models

## Jupyter Noteboooks

[activation_functions.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%204%20-%20Training%20and%20testing%20ANN%20regression%20models/activation_functions.ipynb): defining and plotting the activation functions `sigmoid` and `ReLU`, along with their first derivatives. No additional files needed in the same directory to operate properly.

[train_test_dnn3_regress.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%204%20-%20Training%20and%20testing%20ANN%20regression%20models/train_test_dnn3_regress.ipynb): demonstrating the use of **data_analysis_ES_ANNs.py** module for building and fitting *`Deep Neural Network`* models (with 3 hiiden layers) on our regression data and storaging the results in .pkl files
