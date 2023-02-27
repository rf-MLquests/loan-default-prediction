from Models.random_forest import compressed_random_forest_model
import pandas as pd
from Objects import LoanDefaultPredictionRequest
from Inference.predict import input_as_dataframe, predict


def predict_with_rf(request: LoanDefaultPredictionRequest):
    input_df = input_as_dataframe(request)
    df = pd.read_csv("../loan-default-prediction/Data/hmeq_processed.csv")
    model = compressed_random_forest_model(df)
    return predict(model, input_df)
