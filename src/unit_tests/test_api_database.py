import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import app, get_database, get_predictor, get_producer


class TestApiDatabase(unittest.TestCase):

    def setUp(self) -> None:
        self.predictor = Mock()
        self.database = Mock()
        self.producer = Mock()

        app.dependency_overrides[get_predictor] = lambda: self.predictor
        app.dependency_overrides[get_database] = lambda: self.database
        app.dependency_overrides[get_producer] = lambda: self.producer

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_predict_saves_result_to_kafka(self) -> None:
        features = {
            "variance": 3.6216,
            "skewness": 8.6661,
            "curtosis": -2.8073,
            "entropy": -0.44699,
        }

        self.predictor.predict.return_value = 0

        response = self.client.post(
            "/predict",
            json=features,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "prediction": 0,
                "label": "authentic",
            },
        )

        self.predictor.predict.assert_called_once_with(
            features
        )
        self.producer.send_prediction.assert_called_once()
        sent_message = self.producer.send_prediction.call_args.args[0]

        expected_message_without_time = {
            **features,
            "prediction": 0,
            "label": "authentic",
        }

        for key, value in expected_message_without_time.items():
            self.assertEqual(sent_message[key], value)

        self.assertIn("created_at", sent_message)

        self.database.save_prediction.assert_not_called()

    def test_get_predictions_returns_database_records(
        self,
    ) -> None:
        self.database.get_predictions.return_value = [
            SimpleNamespace(
                id=1,
                variance=3.6216,
                skewness=8.6661,
                curtosis=-2.8073,
                entropy=-0.44699,
                prediction=0,
                label="authentic",
                created_at=datetime(
                    2026, 9, 2, 10, 30, 0
                ),
            )
        ]

        response = self.client.get("/predictions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "id": 1,
                    "variance": 3.6216,
                    "skewness": 8.6661,
                    "curtosis": -2.8073,
                    "entropy": -0.44699,
                    "prediction": 0,
                    "label": "authentic",
                    "created_at": "2026-09-02T10:30:00",
                }
            ],
        )

        self.database.get_predictions.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()