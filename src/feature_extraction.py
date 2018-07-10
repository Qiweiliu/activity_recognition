import numpy as np
import pandas as pd


def extract_velocity_features(data):
    return 1


def extract_acceleration_features(data):
    return 1


def extract_amplitude_features(data):
    return 1


def extract_frequency_features(data):
    return 1


class FeatureExtractor:

    def __init__(self, feature_functions):
        self.feature_functions = feature_functions

    def generate(self, data_sets):
        keys = list(self.feature_functions.keys())
        keys.append('label')
        dt = pd.DataFrame(columns=keys)

        def generate_current_data(data):
            nonlocal dt
            row = pd.Series([np.nan] * len(self.feature_functions),
                            index=self.feature_functions.keys()
                            )
            for feature in self.feature_functions.keys():
                print(row[feature])
                row[feature] = self.feature_functions[feature](data)
            dt = dt.append(row, ignore_index=True)

        for data in data_sets:
            generate_current_data(data)

        return dt
