import pandas as pd
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt

from data_transformation import predictors_one_hot_encoding, log_response, exp_response, scale_numerics

from sklearn.metrics import mean_absolute_error

def transform_validation(training_df, test_df):

    ''' transform to include common factors between train and test '''

    training_df = predictors_one_hot_encoding(training_df)
    test_df = predictors_one_hot_encoding(test_df)

    # As automated, ensure that same cols exist in both
    cols = list(set(list(training_df.columns)) - set(list(test_df.columns)))

    for i in cols:
        test_df[i] = pd.Series([0 for x in range(len(test_df.index))])

    df_cols = list(training_df.columns)
    test_df = test_df[df_cols]

    return test_df

def transform_test(training_df, test_df):

    ''' transform to include common factors between train and test '''

    training_df = predictors_one_hot_encoding(training_df)
    test_df = predictors_one_hot_encoding(test_df)

    # As automated, ensure that same cols exist in both
    cols = list(set(list(training_df.columns)) - set(list(test_df.columns)))

    for i in cols:
        test_df[i] = pd.Series([0 for x in range(len(test_df.index))])

    df_cols = list(training_df.columns)
    df_cols.remove('loss')
    df_cols.remove('id')

    test_df = test_df[df_cols]
    scale_numerics(test_df)

    return test_df

def scores_and_fe(X_train, y_train, training_df, validation_df, model, current_time):

    ''' score the trained model and export feature importance plots '''

    # performance metrics on train
    y_pred=model.predict(X_train)
    print('Training:\t', (mean_absolute_error(y_train, y_pred)))

    # performance metrics on validation
    validation_df = predictors_one_hot_encoding(validation_df)
    validation_df = transform_validation(training_df, validation_df)
    validation_df['loss'] = np.log(validation_df['loss']) #log_response
    scale_numerics(validation_df)
    X_valid = validation_df.drop(['id', 'loss'], axis=1)
    y_valid = validation_df['loss']
    y_test_pred = model.predict(X_valid)
    print('Validation:\t', (mean_absolute_error(y_valid, y_test_pred)))

    # summarize feature importance
    plt.rcParams["figure.figsize"] = (25, 25)
    ax = plot_importance(model)
    ax.figure.tight_layout()
    ax.figure.savefig(f'outputs/feature_importance_{current_time}.png')

def predict_test(training_df, test_df, model, current_time):

    ''' predict on the test dataset for Kaggle upload '''

    # Copy of test_df
    df_test = test_df.copy()

    # Transform test dataset to the same as train
    test_df=transform_test(training_df, test_df)

    # Transform for XGB
    prediction=model.predict(test_df)

    # Aggregate for Kaggle prediction
    y_prediction = pd.DataFrame(prediction)
    y_out = pd.DataFrame(df_test.id)
    y_out = y_out.join(y_prediction)
    y_out.columns=['id','loss']
    exp_response(y_out, 'loss')
    y_out.to_csv(f'outputs/predictions_{current_time}.csv', index=False)