import numpy as np


def object_to_category(df):
    cols = df.select_dtypes(['object']).columns.tolist()
    cols.append('BAD')
    for i in cols:
        df[i] = df[i].astype('category')
    return df


def treat_outliers(data, df, col):
    Q1 = data[col].describe()[4]
    Q3 = data[col].describe()[6]
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)
    return df


def treat_outliers_all(df):
    data = df.copy()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for c in numerical_cols:
        df = treat_outliers(data, df, c)
    return df


def add_missing_val_flag(df, col):
    new_col = str(col)
    new_col += '_missing_values_flag'
    df[new_col] = df[col].isna()
    return df


def add_missing_val_flags(df):
    missing_val_cols = [col for col in df.columns if df[col].isnull().any()]
    for c in missing_val_cols:
        add_missing_val_flag(df, c)


def fillna_with_median(df):
    num_data = df.select_dtypes('number')
    df[num_data.columns] = num_data.fillna(num_data.median())


def fillna_with_mode(df):
    categorical_cols = df.select_dtypes('category').columns.tolist()
    for c in categorical_cols:
        mode = df[c].mode()[0]
        df[c] = df[c].fillna(mode)
