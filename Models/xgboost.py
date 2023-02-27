from xgboost import XGBClassifier
from Training.training import pre_train_prep


def decision_tree_model(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    xgb = XGBClassifier(scale_pos_weight=4, random_state=1, eval_metric='logloss')
    xgb.fit(x_train, y_train)
    return xgb
