import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC


def categorical_to_dummy(DataFrame):
    le = LabelEncoder()
    dataset = DataFrame.copy()
    dataset.dropna(inplace=True, axis=0)
    cat_col = dataset.select_dtypes("object").columns.tolist()

    for column in cat_col:
        if len(dataset.loc[:, column].unique()) <= 2:
            dataset.loc[:, column] = le.fit_transform(dataset.loc[:, column])
            cat_col.remove(column)
    dataset = pd.get_dummies(dataset, columns=cat_col)
    return dataset


def simple_oversampling(DataFrame, dependent_variable):
    """
    Assumes binary class labels and classes are labeled 0 and 1.  
    """
    dataset = DataFrame.copy()
    class_0 = dataset.loc[dataset.loc[:, dependent_variable] == 0, :]
    class_1 = dataset.loc[dataset.loc[:, dependent_variable] == 1, :]
    size_0 = class_0.shape[0]
    size_1 = class_1.shape[0]

    if size_0 > size_1:
        class_1 = class_1.sample(n=size_0, replace=True)
    else:
        class_0 = class_0.sample(n=size_1, replace=True)
    temp = pd.concat([class_0, class_1], axis=0, ignore_index=True)
    return temp


def smotenc_oversampling(DataFrame, y, cat):
    dataset = DataFrame.copy()
    se = SMOTENC(
        categorical_features=cat, k_neighbors=6, n_jobs=10, sampling_strategy="minority"
    )
    resampled_x, resampled_y = se.fit_resample(dataset, y)
    return resampled_x, resampled_y


def get_indicator_columns(DataFrame):
    dataset = DataFrame.copy()
    cols = dataset.columns
    temp = []

    for column in cols:
        if len(dataset.loc[:, column].unique()) == 2:
            idx = dataset.columns.get_loc(column)
            temp.append(idx)
    return temp




