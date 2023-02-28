import pandas as pd
from Objects.response import Response


def input_as_dataframe(request_dict):
    input_df = pd.DataFrame([request_dict])
    input_df = input_df[['LOAN', 'DEBTINC']]
    return input_df


def predict_with_model(model, df):
    if df['DEBTINC'][0] is None:
        df['DEBTINC'] = 34
        df['DEBTINC_missing_values_flag'] = True
    else:
        df['DEBTINC_missing_values_flag'] = False
    print(df)
    return model.predict(df), model.predict_proba(df)


def deserialize_response(labels, probabilities):
    print(labels)
    print(probabilities)
    response = Response(bad=labels[0], goodProb=probabilities[0][0], badProb=probabilities[0][1])
    return response
