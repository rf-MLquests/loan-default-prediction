from Inference.predict import input_as_dataframe, predict_with_model
import pickle


def predict_with_dt(request_dict):
    input_df = input_as_dataframe(request_dict)
    model = pickle.load(open('../loan-default-prediction/Models/decision_tree.pkl', 'rb'))
    return predict_with_model(model, input_df)
