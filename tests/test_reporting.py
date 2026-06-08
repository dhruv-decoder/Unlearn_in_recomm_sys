import os
import tempfile
import unittest

import pandas as pd

from reporting import load_experiment_summary


class ReportingTests(unittest.TestCase):
    def test_load_experiment_summary_reads_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'experiment_results.csv')
            pd.DataFrame([
                {
                    'method': 'kmeans',
                    'num_groups': 2,
                    'baseline_prec@5': 0.1,
                    'naive_unlearn_time': 1.2,
                    'sisa_unlearn_time': 0.4,
                }
            ]).to_csv(path, index=False)

            df = load_experiment_summary(path)
            self.assertEqual(df.loc[0, 'method'], 'kmeans')
            self.assertAlmostEqual(float(df.loc[0, 'baseline_prec@5']), 0.1)
            self.assertAlmostEqual(float(df.loc[0, 'naive_unlearn_time']), 1.2)


if __name__ == '__main__':
    unittest.main()
