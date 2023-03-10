from sklearn.tree import DecisionTreeClassifier
from Training.training import pre_train_prep


def decision_tree_model(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    decision_tree = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    decision_tree.fit(x_train, y_train)
    return decision_tree
