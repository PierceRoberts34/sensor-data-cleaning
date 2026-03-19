# Interpreting Sensor Data with AI Agents
Using open source llms, we can interpret sensor data and create summaries of data. We use prompt engineering to get the best possible results. Rules like "answer in tabular format" allow us to capture the result of the prompt in our code.

# Prerequisites
We need cleaned data to best make use of the agent. Cleaned data helps the AI produce more accurate summaries and more stable results. 
# Usage
Select a time window size. Larger time windows require more memory and risk information being ommitted in the summary, but smaller windows will take additional time to process since the prompt needs to be re-run every time.
