from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

class EnvVars:
    CSV_SENSOR_DATA_PATH = os.getenv('CSV_SENSOR_DATA_PATH')
    PARQUET_SENSOR_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'.parquet'
    ANOMALY_DATA_PATH = os.getenv('ANOMALY_DATA_PATH')
    LLM_MODEL = os.getenv('LLM_MODEL') # Lightweight models are suggested

