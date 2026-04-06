import json
import time
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from ollama import chat, create
from pydantic import BaseModel, ValidationError

from environment import EnvVars

rawdata = EnvVars.CSV_SENSOR_DATA_PATH
metadata_file = EnvVars.SENSOR_METADATA_PATH
appliance_use_data = EnvVars.CSV_APPLIANCE_DATA_PATH
cleaned_data = EnvVars.CLEANED_CSV_DATA_PATH 
parquet_data = EnvVars.PARQUET_SENSOR_DATA_PATH


# Define the output structure with Pydantic
# This restricts the llm's responses
class ResponseFormat(BaseModel):
    flag: Literal['FLAG', 'NO_FLAG']

# Clean the data for analysis
def prepareData():
    # Import data from filesystem and set headers
    colnames = ['status', 'date', 'time', 'sensor']
    df = pd.read_csv(rawdata, header=None, names=colnames)

    # Remove unlabeled sensors (noise) from the dataset
    df = df[df['sensor']!='YYY']

    # Convert to Datetime
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Sorting for Time-Series Analysis
    df = df.sort_values(by=['timestamp', 'sensor'])

    # Fix Sensor Jitter (Debouncing)
    # If a sensor triggers again within 2 seconds, we treat it as noise/redundancy.
    df['time_diff'] = df.groupby('sensor')['timestamp'].diff().dt.total_seconds()
    df = df[~((df['time_diff'] < 2) & (df['time_diff'] >= 0))]
    df = df.drop(columns=['time_diff', 'date', 'time'])

    # Open the file and load its contents
    with open(metadata_file, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    # Set vibration sensor delays based on the metadata
    for sensor, info in metadata['vibration'].items():
        delay = info['Delay']
        df.loc[df['sensor'] == sensor, 'timestamp'] -= pd.Timedelta(seconds=delay)
    
    # Export appliance usage and motion/presence sensors to seperate files
    df[df['sensor'].isin(metadata['appliances'])].to_csv(appliance_use_data)
    df[~df['sensor'].isin(metadata['appliances'])].to_csv(cleaned_data)
    return None

# Assign probabilities to sensor readings using a markov model
def markovProb():
    # Import data from filesystem
    df = pd.read_csv(cleaned_data)

    # Determine next reading
    df['next_sensor'] = df['sensor'].shift(-1)

    # Drop the last reading since it won't have a reading
    df = df.dropna(subset=['next_sensor'])

    # Determine the markov probability
    df['markov_prob'] = df.groupby('sensor')['next_sensor'].transform(
        lambda x: x.map(x.value_counts(normalize=True))
    )

    # Mark values lower than anomaly threshold
    df['anomaly'] = [1 if x <= 0.05 else 0 for x in df['markov_prob']]

    # Save the result
    start_time = time.time()
    df.to_parquet(parquet_data)
    end_time = time.time()
    print(f"DataFrame written to Parquet in {end_time - start_time:.2f} seconds")
    return None

def anomalyFlag(data):
    response = chat('data-agent', 
                    messages=[
                        {'role': 'user', 'content': f"{data}"}
                    ],
                    format=ResponseFormat.model_json_schema())
    return response

def get_decision(row):
    # Only process if it's an anomaly
    if row["anomaly"] == 1:
        try:
            response = anomalyFlag(row)
            decision = ResponseFormat.model_validate_json(response.message.content)
            return decision.flag
        except (ValidationError) as e:
            print(f"Error: {e}")
            return 'FLAG'
    return None

def main():
    # Prepare the data
    prepareData()
    # Compute the markov probabilities
    markovProb()

    # Create a custom model for ease of prompting
    system_prompt = f"""
                    You are a data scientist tasked with interpreting data from sensors placed around a subject's house. 
                    You are tasked with detecting anomalies in the data.
                        Decision rules:
                            - NO_FLAG: noise, ordinary activity
                            - FLAG: severe or ambiguous anomaly
                """
    create(model='data-agent', from_=EnvVars.LLM_MODEL, system=system_prompt)

    db = duckdb.connect()
    db.execute(f"CREATE VIEW subject1 AS SELECT * FROM read_parquet('{parquet_data }')")

    query = f"SELECT * FROM subject1 LIMIT 1000" # Limit results for testing purposes
    df = db.execute(query).df()
    db.close()

    # Apply the function and create/update the column
    start_time = time.time()
    df['decision_flag'] = df.apply(get_decision, axis=1)
    end_time = time.time()
    print(f"All anomalies reviewed in {end_time - start_time:.2f} seconds")

    print(df[df['decision_flag']=='FLAG'])
    df.to_csv(EnvVars.ANOMALY_DATA_PATH)


if __name__ == "__main__":
    main()