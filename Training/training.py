import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle


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


def train_dt_model():
    df = pd.read_csv("../loan-default-prediction/Data/hmeq_processed.csv")
    x_train, x_test, y_train, y_test = pre_train_prep_compressed(df)
    model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    model.fit(x_train, y_train)
    model_file = '../loan-default-prediction/Models/decision_tree.pkl'
    pickle.dump(model, open(model_file, 'wb'))


def train_rf_model():
    df = pd.read_csv("../loan-default-prediction/Data/hmeq_processed.csv")
    x_train, x_test, y_train, y_test = pre_train_prep_compressed(df)
    model = RandomForestClassifier(class_weight='balanced', random_state=1)
    model.fit(x_train, y_train)
    model_file = '../loan-default-prediction/Models/random_forest.pkl'
    pickle.dump(model, open(model_file, 'wb'))


def train_xgb_model():
    df = pd.read_csv("../loan-default-prediction/Data/hmeq_processed.csv")
    x_train, x_test, y_train, y_test = pre_train_prep_compressed(df)
    model = XGBClassifier(scale_pos_weight=4, random_state=1, eval_metric='logloss')
    model.fit(x_train, y_train)
    model_file = '../loan-default-prediction/Models/xgb.pkl'
    pickle.dump(model, open(model_file, 'wb'))
