"""
Development pipeline - simple model to test various aspects of the pipeline
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import config.dev_config as conf
from processing import transformers as tran

multi_label_encoder = tran.ColumnTransformerDF(
        [('LabelEncoder', LabelEncoder(), conf.CATEGORICAL_VARS)],
        remainder='passthrough'
        )

imputer = tran.ColumnTransformerDF([
        ('CategoricalImputer', 
         SimpleImputer(strategy='most_frequent'), 
         conf.CATEGORICAL_VARS
        ),
        ('NumericalImputer',
         SimpleImputer(strategy='mean'),
         conf.NUMERIC_VARS
        )
         ],
        remainder='passthrough'
        )

ohe = ColumnTransformer([
        ('OHE', OneHotEncoder(), conf.CATEGORICAL_VARS)],
        remainder='passthrough'
        )

dev_pipeline = Pipeline([
        ('MultiLabelEncoder', multi_label_encoder),
        ('Imputer', imputer),
        ('TemporalFE', tran.BivariateTransformer(
                variables=conf.TEMPORAL_VARS,
                reference_var=conf.DROP_FEATURES,
                func=lambda a, b: a - b
                )),
        ('DropAfterFE', tran.FeatureDropper(vars_to_drop=conf.DROP_FEATURES)),
        ('RareEncoder', tran.RareLabelEncoder(conf.CATEGORICAL_VARS)),
        ('LogTransform', tran.UnivariateTransformer(
                variables=conf.NUMERICALS_LOG_VARS,
                func=np.log)),
        ('OneHotEncoder', ohe),
        ('LinearModel', LinearRegression())
        ])
