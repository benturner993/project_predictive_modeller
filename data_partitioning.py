import pandas as pd

def training_validation_subset(df):
    ''' function to create training and validation subsets
        chosen this methodology as a method to replicate in the future '''

    training_df = df.sample(frac=0.7)
    print('Training dataset rows:\t', training_df.shape[0])

    validation_df = pd.concat([df, training_df]).drop_duplicates(keep=False)
    print('Validation dataset rows:\t', validation_df.shape[0])

    return training_df, validation_df