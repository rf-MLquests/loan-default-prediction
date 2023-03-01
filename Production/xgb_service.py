from Inference.predict import input_as_dataframe, predict_with_model
import pickle


def predict_with_xgb(request_dict):
    input_df = input_as_dataframe(request_dict)
    model = pickle.load(open('../loan-default-prediction/Models/xgb.pkl', 'rb'))
    print("loaded latest model")
    return predict_with_model(model, input_df)
