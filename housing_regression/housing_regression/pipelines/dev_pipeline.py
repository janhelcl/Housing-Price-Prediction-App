"""
Development pipeline - simple model to test various aspects of the pipeline
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import housing_regression.config.dev_config as conf
from housing_regression.processing import transformers as tran

# different imputing strategy for categorical and numeric
imputer = tran.ColumnTransformerDF(
    [
        (
            "CategoricalImputer",
            SimpleImputer(strategy="most_frequent"),
            conf.CATEGORICAL_VARS,
        ),
        ("NumericalImputer", SimpleImputer(strategy="mean"), conf.NUMERIC_VARS),
    ],
    remainder="passthrough",
)

# FE: time pased between two dates
def diff(a, b):
    return b - a


temporal = tran.BivariateTransformer(
    variables=conf.TEMPORAL_VARS, reference_var=conf.DROP_FEATURES[0], func=diff
)

# log transform selected features
log_tran = tran.UnivariateTransformer(variables=conf.NUMERICALS_LOG_VARS, func=np.log)

# ohe all categorical variables
ohe = ColumnTransformer(
    [("OHE", OneHotEncoder(handle_unknown="ignore"), conf.CATEGORICAL_VARS)],
    remainder="passthrough",
)

dev_pipeline = Pipeline(
    [
        ("Imputer", imputer),
        ("TemporalFE", temporal),
        ("DropAfterFE", tran.FeatureDropper(vars_to_drop=conf.DROP_FEATURES)),
        ("RareEncoder", tran.RareLabelEncoder(conf.CATEGORICAL_VARS)),
        ("LogTransform", log_tran),
        ("OneHotEncoder", ohe),
        ("LinearModel", LinearRegression()),
    ]
)
