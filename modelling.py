import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sklearn2pmml

from skopt import BayesSearchCV
from sklearn.model_selection import KFold

# Hardcoded settings
ITERATIONS = 10 # 1000

# Regressor
bayes_cv_tuner = BayesSearchCV(
    estimator=xgb.XGBRegressor(
        n_jobs=1,
        objective='reg:linear',
        eval_metric='mae',
        silent=1,
        tree_method='approx'
    ),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        #'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100)
        # 'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },
    scoring='neg_mean_absolute_error',
    cv=KFold(n_splits=5,
             random_state=None,
             shuffle=False),
    n_jobs=3,
    n_iter=ITERATIONS,
    verbose=0,
    refit=True,
    random_state=42
)

def status_print(optim_result):

    ''' status callback during bayesian hyperparameter search '''

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest Result: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(f'outputs/' + clf_name + "_cv_results.csv")

def save_model(model, current_time):

    ''' pickle and save as pmml '''

    # pickle
    pickle.dump(model, open(f'outputs/model_{current_time}.sav', 'wb'))

    # pmml
    pmml_object=sklearn2pmml.make_pmml_pipeline(model)
    sklearn2pmml.sklearn2pmml(pmml_object, f'outputs/model_{current_time}.pmml.xml')