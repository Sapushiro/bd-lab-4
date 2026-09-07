import json
import os

from kafka import KafkaProducer

class PredictionProducer:
    def __init__(self) -> None:
        self.topic = os.environ["KAFKA_TOPIC"]

        self.producer = KafkaProducer(
            bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=5
        )

    def send_prediction(self, message: dict) -> tuple[int, int]:
        future = self.producer.send(
            topic=self.topic,
            value=message
        )

        metadata = future.get(timeout=10)

        return metadata.partition, metadata.offset

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()