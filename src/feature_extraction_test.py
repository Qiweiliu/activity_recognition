import unittest
import numpy as np
import src.feature_extraction as fe
import src.signal_process_tools as spt
import pandas as pd

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))

    def test_extract_features(self):
        feature_funcs = {
            'velocity': fe.extract_velocity_features,
            'acceleration': fe.extract_acceleration_features,
            'amplitude': fe.extract_amplitude_features,
            'frequency': fe.extract_frequency_features
        }
        feature_extractor = fe.FeatureExtractor(
            feature_functions=feature_funcs
        )
        print(feature_extractor.generate([1,2,3]))

    def test_learning_pd(self):
        mock_dic = {'1':[],'2':[],'3':[]}
        dt = pd.DataFrame(columns=mock_dic.keys())
        dt.to_csv('test')


if __name__ == '__main__':
    unittest.main()
