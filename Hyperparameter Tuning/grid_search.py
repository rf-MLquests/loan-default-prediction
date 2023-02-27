import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Training.training import pre_train_prep


def dt_grid_search(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    decision_tree_optimized = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    parameters = {
        'max_depth': np.arange(2, 7),
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [5, 10, 20, 25]
    }
    scorer = metrics.make_scorer(recall_score, pos_label=1)
    grid_search = GridSearchCV(decision_tree_optimized, parameters, scoring=scorer, cv=10)
    grid_search = grid_search.fit(x_train, y_train)
    decision_tree_optimized = grid_search.best_estimator_
    decision_tree_optimized.fit(x_train, y_train)
    return decision_tree_optimized


def rf_grid_search(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    random_forest_optimized = RandomForestClassifier(class_weight='balanced', random_state=1)
    parameters = {
        "n_estimators": [100, 250, 500],
        "min_samples_leaf": np.arange(1, 4, 1),
        "max_features": [0.7, 0.9, 'auto'],
    }
    scorer = metrics.make_scorer(recall_score, pos_label=1)
    grid_search = GridSearchCV(random_forest_optimized, parameters, scoring=scorer, cv=10)
    grid_search = grid_search.fit(x_train, y_train)
    random_forest_optimized = grid_search.best_estimator_
    random_forest_optimized.fit(x_train, y_train)


def xgb_grid_search(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    xgboost_optimized = XGBClassifier(scale_pos_weight=4, random_state=1, eval_metric='logloss')
    parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    scorer = metrics.make_scorer(recall_score, pos_label=1)
    grid_search = GridSearchCV(xgboost_optimized, parameters, scoring=scorer, cv=10)
    grid_search = grid_search.fit(x_train, y_train)
    xgboost_optimized = grid_search.best_estimator_
    xgboost_optimized.fit(x_train, y_train)
