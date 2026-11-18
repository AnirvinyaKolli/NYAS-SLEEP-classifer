import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sleep_pipeline import json_safe, parse_label_mapping, validate_prediction_length


class PipelineHelpersTest(unittest.TestCase):
    def test_mapping_is_parsed_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "task-sleep_events.json"
            p.write_text(json.dumps({"Scoring1": {"levels": {
                "s1": "Wake", "s2": "REM", "s3": "N1", "s4": "N2",
                "s5": "N3", "s6": "A", "s7": "Artefact", "s8": "Unscored"}}}))
            mapping, _ = parse_label_mapping(p)
            self.assertEqual(mapping, {1: "Wake", 2: "REM", 3: "N1", 4: "N2",
                                       5: "N3", 6: "A", 7: "Artefact", 8: "Unscored"})

    def test_json_safe_converts_numpy_scalars_and_keys(self):
        obj = json_safe({np.int64(4): np.float32(1.25), "values": np.array([np.int64(2)])})
        self.assertEqual(obj, {"4": 1.25, "values": [2]})
        json.dumps(obj, allow_nan=False)

    def test_prediction_length_matches_labels(self):
        validate_prediction_length(np.array(["W", "N2"]), 2, "test")
        with self.assertRaises(AssertionError):
            validate_prediction_length(np.array(["W"]), 2, "test")


if __name__ == "__main__":
    unittest.main()
