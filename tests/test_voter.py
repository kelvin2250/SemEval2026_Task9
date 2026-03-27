import os
import tempfile
import unittest

import pandas as pd

from ensemble.ensemble import weighted_soft_voting


class TestVoter(unittest.TestCase):
    def test_weighted_soft_voting_outputs_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "model1.csv")
            file2 = os.path.join(tmpdir, "model2.csv")
            output_pred = os.path.join(tmpdir, "pred.csv")
            output_prob = os.path.join(tmpdir, "prob.csv")

            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "prob_class1": [0.9, 0.2],
                }
            ).to_csv(file1, index=False)

            pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "prob_class1": [0.3, 0.8],
                }
            ).to_csv(file2, index=False)

            weighted_soft_voting(
                files_with_weights=[(file1, 0.75), (file2, 0.25)],
                output_pred=output_pred,
                output_prob=output_prob,
                threshold=0.5,
            )

            pred_df = pd.read_csv(output_pred, dtype={"id": str})
            prob_df = pd.read_csv(output_prob, dtype={"id": str})

            self.assertEqual(pred_df.loc[pred_df["id"] == "1", "polarization"].iloc[0], 1)
            self.assertEqual(pred_df.loc[pred_df["id"] == "2", "polarization"].iloc[0], 0)

            prob_map = {str(row["id"]): row["prob_class1"] for _, row in prob_df.iterrows()}
            self.assertAlmostEqual(prob_map["1"], 0.75, places=6)
            self.assertAlmostEqual(prob_map["2"], 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
