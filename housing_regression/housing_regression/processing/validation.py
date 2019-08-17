"""
Functionality to validate the imput data

# TODO: extend to allow different pipes have different conditions
# TODO: report which rows where filtered
"""
import pandas as pd

import housing_regression.config.dev_config as conf


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows with missing values where not expected
    
    :param input_data: dataframe of data to be filtered
    
    :returns: dataframe with filtered rows
    """

    validated_data = input_data.copy()

    if input_data[conf.NAN_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=conf.NAN_NOT_ALLOWED)
    
    return validated_data
