import pandas as pd
from Objects import LoanDefaultPredictionRequest


def input_as_dataframe(request: LoanDefaultPredictionRequest):
    upper_case_dict = {k.upper(): v for k, v in request.dict().items()}
    input_df = pd.DataFrame.from_dict(upper_case_dict, orient='columns')
    print(input_df)
    return input_df


def predict(model, df):
    input = df.copy()
    if input['DEBTINC']:
        input['DEBTINC_missing_values_flag'] = False
    else:
        input['DEBTINC'] = 34
        input['DEBTINC_missing_values_flag'] = True
    print(input)
    return model.predict(input)
