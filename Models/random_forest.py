from sklearn.ensemble import RandomForestClassifier
from Training.training import pre_train_prep, pre_train_prep_compressed


def random_forest_model(df):
    x_train, x_test, y_train, y_test = pre_train_prep(df)
    random_forest = RandomForestClassifier(class_weight='balanced', random_state=1)
    random_forest.fit(x_train, y_train)
    return random_forest


def compressed_random_forest_model(df):
    x_train, x_test, y_train, y_test = pre_train_prep_compressed(df)
    random_forest = RandomForestClassifier(class_weight='balanced', random_state=1)
    random_forest.fit(x_train, y_train)
    return random_forest
