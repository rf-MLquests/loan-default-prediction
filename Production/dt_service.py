from Models.decision_tree import compressed_decision_tree_model
import pandas as pd
from Inference.predict import input_as_dataframe, predict_with_model


def predict_with_dt(request_dict):
    input_df = input_as_dataframe(request_dict)
    df = pd.read_csv("../loan-default-prediction/Data/hmeq_processed.csv")
    model = compressed_decision_tree_model(df)
    return predict_with_model(model, input_df)
