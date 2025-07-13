# Part 3 - Training and testing machine learning regression models

In this directory we present our codes for building and evaluating machine learning regression models, using our regression data. For more about how these data where produced, see [Part 1](https://github.com/istergak/MSc-Computational-Physics-AUTH/tree/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%201%20-%20Solving%20the%20TOV%20equations%20for%20Hadronic%20and%20Quark%20Stars) and [Part 2](https://github.com/istergak/MSc-Computational-Physics-AUTH/tree/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%202%20-%20Handling%20the%20TOV%20equations%20solution%20data%20for%20Exotic%20Stars).

**Algorithms**

We used the following machine learning algorithms:<br>
->`DecisionTreeRegressor` (see [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html))<br>
->`RandomForestRegressor` (see [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))<br>
->`GradientBoostingRegressor` (see [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html))<br>
->`XGBRegressor` (see [documentation](https://xgboost.readthedocs.io/en/stable/parameter.html))

**Notes**

1. Files of regression data and summary results must be included in the same directory with the following modules and notebooks, for the algorithms to run properly.

2. For all models, we performed **5-fold cross-validation** combined with **Grid search** for fine-tuning

**Links to regression data and summary results**

->Regression data: [here](https://drive.google.com/drive/folders/1eFYPW1juSy4aSwTBDs-ye0ToogRbSJfv)<br>
->Summary results: [here](https://drive.google.com/drive/folders/1bNaBKrXdxmViiz2ZDXu0dWDxsOv0pNsL)

## Modules

[data_analysis_ES_ML.py](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/data_analysis_ES_ML.py):Module containing functions and classes for **a)** assessing linear correlations in regression data, **b)** training and testing machine learning regression models, **c)** storaging the fitting results in .pkl files and loading the .pkl files of trained models, and **d)** storaging summary results in .csv files, loading and presenting the summary results in *PrettyTable* and *bar plots* forms

## Jupyter Notebooks

[assessing_regression_data.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/assessing_regression_data.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for assessing linear correlations in our regression data

[train_test_dtree_regress.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/train_test_dtree_regress.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for fitting *`Decision Tree`* models on our regression data and storaging the results on .pkl files

[train_test_rf_regress.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/train_test_rf_regress.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for fitting *`Random Forest`* models on our regression data and storaging the results on .pkl files

[train_test_gradboost_regress.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/train_test_gradboost_regress.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for fitting *`Gradient Boosting`* models on our regression data and storaging the results on .pkl files

[train_test_xgboost_regress.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/train_test_xgboost_regress.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for fitting *`XGBoost`* models on our regression data and storaging the results on .pkl files

[assessing_summary_ml_reg.ipynb](https://github.com/istergak/MSc-Computational-Physics-AUTH/blob/main/Thesis%20-%20ML%20and%20ANNs%20regression%20models%20for%20Exotic%20Star's%20EOSs/Part%203%20-%20Training%20and%20testing%20ML%20regression%20algorithms/assessing_summary_ml_reg.ipynb): demonstrating the use of **data_analysis_ES_ML.py** module for loading the .pkl files of trained models, storaging summary results in .csv files, loading and presenting the summary results in *PrettyTable* and *bar plots* forms
