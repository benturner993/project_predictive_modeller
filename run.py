from time import strftime, gmtime

from data import load_data
from data_partitioning import training_validation_subset
from data_transformation import response_outlier_capping, log_response, predictors_one_hot_encoding
from modelling import bayes_cv_tuner, status_print, save_model
from prediction import scores_and_fe, predict_test

# hardcoded settings
current_time = strftime("%Y%m%d%H%M%S", gmtime())

# load data
training_df, test_df = load_data()
print('dataset loaded...')

# data partitioning
training_df, validation_df=training_validation_subset(training_df)
print('dataset partitioned...')

# encode features
training_df=predictors_one_hot_encoding(training_df)
training_df=response_outlier_capping(training_df, 'loss', 2.2)
training_df=log_response(training_df, 'loss')
print('dataset encoded...')

# feature set
X=training_df.drop(['id', 'loss'], axis=1)
y=training_df['loss']
print('feature set selected...')

# fit (and save) the model
xgb_bo = bayes_cv_tuner.fit(X, y, callback=status_print)
save_model(xgb_bo.best_estimator_, current_time)
print('modelling complete...')

# scores and feature importance
scores_and_fe(X, y, training_df, validation_df, xgb_bo.best_estimator_, current_time)
print('modelling scores created...')

# predict on test dataset
predict_test(training_df, test_df, xgb_bo.best_estimator_, current_time)
print('predictions complete...')

#Notes:
# response variable: further investigate skewing using tukeys outliers
# look at residuals after model build and find if normally distributed
# box cox transform, optimal to power to transform by (transform data thats postivie)
#scaling
#null values
#dataset partititions on train
#modelling
#algorithm
#hyperopt
#stacking
#predictions
#save model
#boxcox
#interpretation - partial dependence plots
# add logger
# pmml