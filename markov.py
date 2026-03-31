import duckdb
import io
import time
from ollama import chat, create
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from typing import Literal

from environment import EnvVars

rawdata = EnvVars.CSV_SENSOR_DATA_PATH
parquet_data = EnvVars.PARQUET_SENSOR_DATA_PATH

# Define the output structure with Pydantic
# This restricts the llm's responses
class ResponseFormat(BaseModel):
    flag: Literal['FLAG', 'NO_FLAG']

def prepareData():
    colnames = ['status', 'date', 'time', 'sensor']
    df = pd.read_csv(rawdata, header=None, names=colnames)

    # We will clean the data and assign probabilities to sensor readings using a markov model

    # Remove unlabeled sensors (noise) from the dataset
    df = df[df['sensor']!='YYY']

    # Remove the first 7 rows since their time is out-of-sequence
    df = df.iloc[7:]

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