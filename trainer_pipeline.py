from src.components.dataIngestion import DataIngestion
from src.components.dataPreprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run_pipeline(self):
        Ingestion=DataIngestion()
        data=Ingestion.data_ingest(r"C:\Users\DELL\OneDrive\Desktop\Diabetes_prediction_model\data\diabetes.csv")

        print("Data Ingestion Complete")

        preprocessing=DataPreprocessing()
        x,y,transformer=preprocessing.preprocess_data(data)

        print("Preprocessing Complete")

        trainer = ModelTrainer()
        trainer.TrainModel(x, y, transformer)

        print("Model Training Completed")

if __name__=="__main__":

    pipeline=TrainPipeline()
    pipeline.run_pipeline()
