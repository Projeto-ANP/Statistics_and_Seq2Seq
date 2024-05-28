import pandas as pd
import numpy as np
from statistics import median
from statistics import mean
class VotingCombination:
    def __init__(self, estimators, combination='mean'):
        self.estimators = estimators
        self.combination = combination
    
    def update(self, estimators, combination):
        self.estimators = estimators
        self.combination = combination
    
    def predict(self, test=None):
        predictions = []
        # for estimator_name, estimator_predictions in self.estimators:
        #     predictions.append(estimator_predictions)            
        if self.combination == 'mean':
            math_func = mean
        elif self.combination == 'median':
            math_func = median
        elif self.combination == 'oracle':
            if test is None:
                raise ValueError("Parameter 'test' is required for combination method 'oracle'.")
            
            combined_predictions = []
            
            for i in range(len(test)):
                distances = [abs(estimator_predictions[i] - test[i]) for _, estimator_predictions in self.estimators]
                closest_index = np.argmin(distances)
                combined_predictions.append(self.estimators[closest_index][1][i])
            combined_predictions = pd.Series(combined_predictions, index=sformat.index)
            return combined_predictions
        else:
            raise ValueError("Unsupported combination method. Please choose 'mean' or 'oracle'.")
        

        sformat = next(iter(self.estimators.values()))

        num_elements = len(sformat)    
        combined_predictions = []
        lista_preds = []
        for i in range(num_elements):
            values_at_index = [self.estimators[format][i] for format in self.estimators.keys()]
            
            index_mean = math_func(values_at_index)
            lista_preds.append(index_mean)
        combined_predictions = pd.Series(lista_preds, index=sformat.index)

        return combined_predictions
