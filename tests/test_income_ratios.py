import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from modeling.experiments.catboost_income_ratios import add_ratios, choose_threshold


class IncomeRatioExperimentTests(unittest.TestCase):
    def test_invalid_income_is_unknown_and_input_is_preserved(self):
        frame = pd.DataFrame({
            'AMT_INCOME_TOTAL': [100., 0., -1., np.nan],
            'AMT_CREDIT': [200.] * 4,
            'AMT_ANNUITY': [10.] * 4,
        })
        original = frame.copy(deep=True)
        result = add_ratios(frame)
        self.assertEqual(result.loc[0, 'APP_CREDIT_TO_INCOME'], 2.)
        self.assertEqual(result.loc[0, 'APP_ANNUITY_TO_INCOME'], .1)
        self.assertTrue(result.iloc[1:][[
            'APP_CREDIT_TO_INCOME', 'APP_ANNUITY_TO_INCOME',
        ]].isna().all().all())
        pd.testing.assert_frame_equal(frame, original)

    def test_threshold_maximizes_observed_f1(self):
        actual = np.array([0, 1, 0, 1, 0, 1])
        probability = np.array([.1, .3, .2, .4, .6, .8])
        threshold = choose_threshold(actual, probability)
        best = max(f1_score(actual, probability >= t) for t in probability)
        self.assertEqual(f1_score(actual, probability >= threshold), best)


if __name__ == '__main__':
    unittest.main()
