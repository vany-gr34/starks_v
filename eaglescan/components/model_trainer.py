import os,sys
import yaml
from eaglescan.utils.main_utils import read_yaml_file
from eaglescan.logger import logging
from eaglescan.exception import AppException
from eaglescan.entity.config_entity import ModelTrainerConfig
from eaglescan.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO
from eaglescan.entity.artifacts_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
)


class ModelTrainer:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise AppException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer")

        try:
          #1. trying to find the data.yaml
            data_yaml_path = os.path.join(
                self.data_ingestion_artifact.feature_store_file_path,
                "data.yaml",
            )

            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError("data.yaml not found in feature store")

            # 2. Load pretrained YOLOv11 model
            logging.info(
                f"Loading YOLO model: {self.model_trainer_config.weight_name}"
            )
            model = YOLO(self.model_trainer_config.weight_name)

            # 3. Train the model
            logging.info("Starting YOLOv11 training")
            results = model.train(
                data=data_yaml_path,
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                project=self.model_trainer_config.model_trainer_dir,
                name="train",
                exist_ok=True,
            )

            # 4. Locate trained model
            best_model_path = os.path.join(
                results.save_dir, "weights", "best.pt"
            )

            if not os.path.exists(best_model_path):
                raise FileNotFoundError("best.pt not found after training")

            logging.info(f"Training completed. Best model at {best_model_path}")

            # 5. Return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=best_model_path
            )

            logging.info(
                f"Exited initiate_model_trainer with artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)


    

