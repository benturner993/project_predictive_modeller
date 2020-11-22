import pandas as pd
import numpy as np
import math

def response_outlier_capping(df, variable, multiplier):

    ''' windsorise the response variable '''

    q1 = np.percentile(df[variable],25)
    q3 = np.percentile(df[variable],75)
    iqr = q3 - q1
    lower = q1 - (iqr * multiplier)
    upper = q3 + (iqr * multiplier)

    df[variable] = np.where(df[variable]<=lower, lower, df[variable])
    df[variable] = np.where(df[variable]>=upper, upper, df[variable])

    return df

def log_response(df, response):

    ''' take the natural log of the response variable '''

    print('Skewness of untransformed response:\t' + str(df[response].skew()))

    # transform response column to ensure +ve
    minimum_val = math.ceil(min(abs(np.log(df[response]))))
    original_data = np.log(df[response]) + minimum_val
    df[response] = np.log(df[response])
    print('Skewness of transformed response:\t' + str(df[response].skew()))

    return df

def predictors_one_hot_encoding(df):

    ''' one hot encode categorical features '''

    # find all relevant columns
    all_columns = list(df.columns)
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
    numeric_columns = df.select_dtypes(include=numeric_types).columns.to_list()
    categoric_columns = list(set(all_columns) - set(numeric_columns))

    for i in categoric_columns:
        one_hot = pd.get_dummies(df[i], prefix=i)
        df = df.join(one_hot)

    # remove categoric cols
    numeric_columns = df.select_dtypes(include=numeric_types).columns.to_list()
    df = df[numeric_columns]

    return df

def exp_response(df, response):

    ''' transform the response variable back to original'''

    df[response] = np.exp(df[response])


def find_numerics(df):
    ''' searches a dataframe and returns numeric columns (excluding id cols)'''

    # numeric data types
    numeric_dtype = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # define all numeric cols
    all_numeric_columns = list(df.select_dtypes(include=numeric_dtype).columns)
    id_columns = ['id', 'loss']

    # define all numeric cols excluding id cols
    numeric_columns = list(set(all_numeric_columns) - set(id_columns))
    return numeric_columns


def store_scaling_values(df):
    ''' stores mean and std values '''

    # load numeric features to be scaled
    numeric_columns=find_numerics(df)

    # create dataframe with stored values
    scaling_df = pd.DataFrame([p, df[p].mean(), df[p].std()] for p in numeric_columns)
    scaling_df.columns = ['col', 'mean', 'std']

    # save the values
    scaling_df.to_csv('outputs/scaling.csv', index=False)


def scale_numerics(df):

    ''' read the scaled values and convert numeric features '''

    # read the scaled values
    scaling_df = pd.read_csv('outputs/scaling.csv')

    # scale columns
    for i in range(scaling_df.shape[0]):
        col = scaling_df.iloc[i][0]
        mean = scaling_df.iloc[i][1]
        std = scaling_df.iloc[i][2]
        df[col] = (df[col] - mean) / std

