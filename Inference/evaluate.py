import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Eligible', 'Not Eligible'],
                yticklabels=['Eligible', 'Not Eligible'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def evaluate_on_test(model, x_test, y_test):
    y_test_pred = model.predict(x_test)
    metrics_score(y_test, y_test_pred)


def show_feature_importance(model, df):
    X = df.drop(["BAD"], axis=1)
    importances = model.feature_importances_
    columns = X.columns
    importance_df = pd.DataFrame(importances, index=columns, columns=['Importance']).sort_values(by='Importance',
                                                                                                 ascending=False)
    plt.figure(figsize=(13, 13))
    sns.barplot(importance_df.Importance, importance_df.index)
