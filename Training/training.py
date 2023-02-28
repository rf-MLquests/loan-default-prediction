import pandas as pd
from sklearn.model_selection import train_test_split


def pre_train_prep(df):
    X = df.drop(["BAD"], axis=1)
    X = pd.get_dummies(data=X, columns=["REASON", "JOB"], drop_first=True)
    y = df["BAD"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def pre_train_prep_compressed(df):
    X = df[['LOAN', 'DEBTINC', 'DEBTINC_missing_values_flag']]
    y = df["BAD"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test
