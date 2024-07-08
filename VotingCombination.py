import pandas as pd
import numpy as np
from statistics import median
from statistics import mean
class VotingCombination:
    def __init__(self, estimators, combination='mean'):
        if len(estimators) < 2:
            raise ValueError("At least 2 regressors as estimators is required")
        self.estimators = estimators
        self.estimators_val = None
        self.combination = combination
        self.moe = None
    
    def update(self, estimators, combination):
        self.estimators = estimators
        self.combination = combination
    
    def update_moe(self, x):
        self.moe = x
    def fit_moe(self, estimators_val, test_val):
        experts = []
        if len(estimators_val) < 2:
            raise ValueError("At least 2 regressors as estimators is required")
        self.estimators_val = estimators_val
        for i in range(len(test_val)):
            distances = {key: abs(predictions[i] - test_val[i]) for key, predictions in self.estimators_val.items()}
            closest_key = min(distances, key=distances.get)
            experts.append((i, closest_key))
        self.moe = experts
    def get_moe(self):
        return self.moe
    def predict(self, test=None):
        sformat = next(iter(self.estimators.values()))
        # for estimator_name, estimator_predictions in self.estimators:
        #     predictions.append(estimator_predictions)            
        if self.combination == 'mean':
            math_func = mean
        elif self.combination == 'median':
            math_func = median
        elif self.combination == 'max':
            # math_func = max
            num_elements = len(sformat)
            combined_predictions = []
            lista_preds = []
            
            for i in range(num_elements):
                values_at_index = [self.estimators[format][i] for format in self.estimators.keys()]
                
                index_max = max(values_at_index)
                lista_preds.append(index_max)
            
            combined_predictions = pd.Series(lista_preds, index=sformat.index)
            
            return combined_predictions
        elif self.combination == 'min':
            math_func = min
        elif self.combination == 'oracle':
            if test is None:
                raise ValueError("Parameter 'test' is required for combination method 'oracle'.")
            
            combined_predictions = []
            
            for i in range(len(test)):
                distances = [abs(estimator_predictions[i] - test[i]) for _, estimator_predictions in self.estimators.items()]
                closest_index = np.argmin(distances)
                combined_predictions.append(self.estimators[closest_index][1][i])
            combined_predictions = pd.Series(combined_predictions, index=sformat.index)
            return combined_predictions
        elif self.combination == "moe":
            if self.moe == None:
                raise ValueError("You need to fit_moe with validation data first..")
            
            combined_predictions = []
            for h, regr in self.moe:
                combined_predictions.append(self.estimators[regr][h])
            
            combined_predictions = pd.Series(combined_predictions, index=sformat.index)
            return combined_predictions

        else:
            raise ValueError("Unsupported combination method. Please choose 'mean' or 'oracle'.")

        num_elements = len(sformat)    
        combined_predictions = []
        lista_preds = []
        for i in range(num_elements):
            values_at_index = [self.estimators[format][i] for format in self.estimators.keys()]
            
            index_mean = math_func(values_at_index)
            lista_preds.append(index_mean)
        combined_predictions = pd.Series(lista_preds, index=sformat.index)

        return combined_predictions
