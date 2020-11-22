import pandas as pd

def load_data():

    ''' read train and test datasets '''

    training_df = pd.read_csv('data/train.csv')[:100]
    test_df=pd.read_csv('data/test.csv')

    return training_df, test_df