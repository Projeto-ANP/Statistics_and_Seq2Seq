from . import aux

import pandas as pd
     
def median_combination(models, dataset_name, dataset_index, val_type="final"):
    # grab the full structure from the aux module
    model_preds = aux.get_predictions_models(models, dataset_index, dataset_name)

    # make a DataFrame whose columns are models and rows are time steps
    df = pd.DataFrame({m: model_preds[m][val_type]["predictions"] for m in models})
    test_series = model_preds[models[0]][val_type]["test"] 

    mean_preds = df.median(axis=1)
    return mean_preds, test_series, model_preds[models[0]][val_type]["start_test"], model_preds[models[0]][val_type]["final_test"]

#python -m combinations.median
if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "svr",
        "rf",
        "catboost",
        "CWT_svr",
        "DWT_svr",
        "FT_svr",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_CWT_svr",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_DWT_svr",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "ONLY_FT_svr",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]
    
    dataset_name = "ANP_MONTHLY"
    
    len_datasets = aux.get_dataset_size(models[0], dataset_name=dataset_name)
    for dataset_index in range(len_datasets):
        mean_result, test_values, start_test, final_test = median_combination(models, dataset_name=dataset_name, dataset_index=dataset_index)
        aux.save_to_csv(exp_name="median", predictions=mean_result, test_values=test_values, dataset_name=dataset_name, dataset_index=dataset_index, horizon=12, start_test=start_test, final_test=final_test)