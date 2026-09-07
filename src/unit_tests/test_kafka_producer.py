import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.kafka_producer import PredictionProducer

class TestPredictionProducer(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction_producer = PredictionProducer.__new__(PredictionProducer)
        self.prediction_producer.topic = "banknote-predictions"
        self.prediction_producer.producer = Mock()

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

    def test_send_prediction_sends_message_to_topic(self) -> None:
        future = Mock()
        future.get.return_value = SimpleNamespace(
            partition=0,
            offset=5
        )

        self.prediction_producer.producer.send.return_value = future
        self.prediction_producer.send_prediction(self.message)
        self.prediction_producer.producer.send.assert_called_once_with(
            topic="banknote-predictions",
            value=self.message
        )

    def test_send_prediction_waits_for_confirmation(self) -> None:
        future = Mock()
        future.get.return_value = SimpleNamespace(
            partition=0,
            offset=5
        )

        self.prediction_producer.producer.send.return_value = future
        self.prediction_producer.send_prediction(self.message)
        future.get.assert_called_once_with(timeout=10)

    def test_send_prediction_returns_partition_and_offset(self) -> None:
        future = Mock()
        future.get.return_value = SimpleNamespace(
            partition=2,
            offset=15
        )

        self.prediction_producer.producer.send.return_value = future
        result = self.prediction_producer.send_prediction(self.message)

        self.assertEqual(result, (2, 15))

    def test_close_flushes_and_closes_producer(self) -> None:
        self.prediction_producer.close()

        self.prediction_producer.producer.flush.assert_called_once_with()
        self.prediction_producer.producer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

