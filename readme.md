![predictive](misc/predictive_modeller.png)

## Objective

Automatation of predictive modelling using XGBoost and Bayesian Optimisation.

## Contents
This repository contains:

- *data.py* - loads in train and test datasets
- *data_partitioning.py* - creates training/validation/holdout and tracable cross validation
- *data_transformation.py* - transformation of response and predictors (including one-hot encoding)
- *modelling.py* - creates model using xgboost and optimises hyper-parameters using Bayesian Optimisation
- *prediction.py* - functions to score the model and visualise outputs
- *run.py* - orchestration file to automate model builds
- *requirements.txt* - list of packages used

## Future Developments
Developments to investigate / consider:

- response variable transformations (e.g. tukeys outliers, box-cox)
- assess benefit added from feature scaling
- increase complexity of feature selection
- add function to handle null values and fairly treat MI
- enable model stacking to ensure model stability (perhaps too complex)
- interpretation layer (e.g. partial dependence plots, shap, lime)
- check radar interpretation using pmml
- continue to read kaggle comp solutions

## Author
Ben Turner
Q4 2020