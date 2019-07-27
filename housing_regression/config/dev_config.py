"""
Configuration of a simple model used to test various aspects of the pipeline
"""
SEED = 42

# all variables used in the pipeline
FEATURES = [# predictors
            'GrLivArea', # numeric to transform
            'YearRemodAdd', # to test FE in pipeline
            'LotFrontage', # numeric with NaNs - to test numeric imputing
            'GarageFinish', # cat with NaNs - to test categorical imputing
            'Utilities', # cat with rare category to test RareLabelEncoder
            # for feature engineering - drop afterwards
            'YrSold']

# categorical variables
CATEGORICAL_VARS = ['GarageFinish', 'Utilities']
# all other numeric
NUMERIC_VARS = [var for var in FEATURES if var not in CATEGORICAL_VARS]

# to test FE in pipeline
TEMPORAL_VARS = ['YearRemodAdd']
DROP_FEATURES = 'YrSold'

# variables to log transform
NUMERICALS_LOG_VARS = ['GrLivArea']