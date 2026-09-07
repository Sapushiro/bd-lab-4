import os
import time
import unittest

import httpx

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)

POLL_TIMEOUT_SECONDS = 20
POLL_INTERVAL_SECONDS = 0.5

class TestDatabase(unittest.TestCase):
    def test_prediction_is_saved_to_database(self) -> None:
        features = {
            "variance": 1.234567,
            "skewness": -2.345678,
            "curtosis": 3.456789,
            "entropy": -4.567891,
        }

        history_before_response = httpx.get(
            f"{API_URL}/predictions",
            timeout=10.0
        )
        self.assertEqual(history_before_response.status_code, 200)

        ids_before = {record["id"] for record in history_before_response.json()}

        prediction_response = httpx.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10.0
        )
        self.assertEqual(prediction_response.status_code, 200)
        prediction_result = prediction_response.json()

        saved_record = self.wait_for_saved_prediction(
            ids_before=ids_before,
            features=features,
            prediction_result=prediction_result
        )

        self.assertAlmostEqual(saved_record["variance"], features["variance"])
        self.assertAlmostEqual(saved_record["skewness"], features["skewness"])
        self.assertAlmostEqual(saved_record["curtosis"], features["curtosis"])
        self.assertAlmostEqual(saved_record["entropy"], features["entropy"])
        self.assertEqual(saved_record["prediction"], prediction_result["prediction"])
        self.assertEqual(saved_record["label"], prediction_result["label"])
        self.assertIsNotNone(saved_record["created_at"])

    def wait_for_saved_prediction(
            self,
            ids_before: set[int],
            features: dict[str, float],
            prediction_result: dict
    ) -> dict:
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            history_response = httpx.get(
                f"{API_URL}/predictions",
                timeout=10.0
            )

            self.assertEqual(history_response.status_code, 200)

            new_records = [record for record in history_response.json() if record["id"] not in ids_before]

            for record in new_records:
                if self.is_expected_prediction(record, features, prediction_result):
                    return record

            time.sleep(POLL_INTERVAL_SECONDS)

        self.fail(
            "Prediction was not saved by Kafka Consumer "
            f"within {POLL_TIMEOUT_SECONDS} seconds"
        )

    @staticmethod
    def is_expected_prediction(
            record: dict,
            features: dict[str, float],
            prediction_result: dict
    ) -> bool:
        return (
                record["variance"] == features["variance"]
                and record["skewness"] == features["skewness"]
                and record["curtosis"] == features["curtosis"]
                and record["entropy"] == features["entropy"]
                and record["prediction"]
                == prediction_result["prediction"]
                and record["label"]
                == prediction_result["label"]
        )


if __name__ == "__main__":
    unittest.main()

