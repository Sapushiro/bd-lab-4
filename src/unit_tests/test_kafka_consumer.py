import unittest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from src.kafka_consumer import PredictionConsumer

class TestPredictionConsumer(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction_consumer = PredictionConsumer.__new__(PredictionConsumer)

        self.prediction_consumer.topic = "banknote-predictions"

        self.prediction_consumer.consumer = MagicMock()
        self.prediction_consumer.database = Mock()

        self.message = {
            "variance": 3.6216,
            "skewness": 8.6661,
            "curtosis": -2.8073,
            "entropy": -0.44699,
            "prediction": 0,
            "label": "authentic",
            "created_at": (
                "2026-09-07T12:30:00+00:00"
            ),
        }

    def test_process_message_save_prediction_to_database(self) -> None:
        self.prediction_consumer.database.save_prediction.return_value = 1
        result = self.prediction_consumer.process_message(self.message)

        self.prediction_consumer.database.save_prediction.assert_called_once_with(
            features={
                "variance": 3.6216,
                "skewness": 8.6661,
                "curtosis": -2.8073,
                "entropy": -0.44699,
            },
            prediction=0,
            label="authentic",
        )

        self.assertEqual(result, 1)

    def test_run_commits_offset_after_successfully_save(self) -> None:
        record = SimpleNamespace(
            value = self.message,
            partition=0,
            offset=5
        )

        self.prediction_consumer.consumer.__iter__.return_value = iter([record])
        self.prediction_consumer.database.save_prediction.return_value = 1

        self.prediction_consumer.run()
        self.prediction_consumer.consumer.commit.assert_called_once_with()
        self.prediction_consumer.consumer.close.assert_called_once_with()

    def test_run_commits_invalid_message(self) -> None:
        invalid_message = {
            "prediction": 0,
            "label": "authentic"
        }

        record = SimpleNamespace(
            value=invalid_message,
            partition=0,
            offset=5
        )

        self.prediction_consumer.consumer.__iter__.return_value = iter([record])
        self.prediction_consumer.run()

        self.prediction_consumer.database.save_prediction.assert_not_called()
        self.prediction_consumer.consumer.commit.assert_called_once_with()
        self.prediction_consumer.consumer.close.assert_called_once_with()

    def test_run_does_not_commit_when_database_fails(self) -> None:
        record = SimpleNamespace(
            value=self.message,
            partition=0,
            offset=5
        )

        self.prediction_consumer.consumer.__iter__.return_value = iter([record])

        self.prediction_consumer.database.save_prediction.side_effect = SQLAlchemyError("Database is unavailable")

        with self.assertRaises(SQLAlchemyError):
            self.prediction_consumer.run()

        self.prediction_consumer.consumer.commit.assert_not_called()
        self.prediction_consumer.consumer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
