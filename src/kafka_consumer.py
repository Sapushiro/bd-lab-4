import json
import os
import logging

from kafka import KafkaConsumer
from src.database import Database

from kafka.errors import KafkaError
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

class PredictionConsumer:
    def __init__(self) -> None:
        self.topic = os.environ["KAFKA_TOPIC"]

        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
            group_id=os.environ["KAFKA_CONSUMER_GROUP"],
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda value: json.loads(value.decode("utf-8"))
        )

        self.database = Database()
        self.database.initialize()

    def process_message(self, message: dict) -> int:
        features = {
            "variance": message["variance"],
            "skewness": message["skewness"],
            "curtosis": message["curtosis"],
            "entropy": message["entropy"]
        }

        return self.database.save_prediction(
            features=features,
            prediction=message["prediction"],
            label=message["label"]
        )

    def run(self) -> None:
        logger.info("Consumer started. Topic: %s", self.topic)

        try:
            for record in self.consumer:
                logger.info("Received message: partition=%s, offset=%s", record.partition, record.offset)

                try:
                    prediction_id = self.process_message(record.value)

                except (KeyError, TypeError, ValueError):
                    logger.exception("Invalid prediction message: %s", record.value)
                    self.consumer.commit()
                    continue

                except SQLAlchemyError:
                    logger.exception(
                        "Could not save prediction to database. "
                        "Offset will not be committed."
                    )
                    raise

                try:
                    self.consumer.commit()
                except KafkaError:
                    logger.exception(
                        "Prediction was saved, but Kafka offset could not be commited."
                    )
                    raise

                logger.info("Prediction saved: id=%s, "
                            "partition=%s, offset=%s",
                            prediction_id,
                            record.partition,
                            record.offset
                )
        finally:
            self.consumer.close()
            logger.info("Consumer stopped")


if __name__ == "__main__":
    PredictionConsumer().run()
