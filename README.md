# Multimodal Anomaly Detection with LLMs
By combining a relatively simple anomaly detection method, with a lightweight large language model, we can create an anomaly detection system which helps reduce the odds of a false positive. Multimodal detection also reduces computational overhead as the data is reduced by the more rapid first detection pass. A multimodal system works well with live deployments and continuous data streams.

Markov was selected as it is better suited to anomaly detection in timeseries data compared to isolation forest, since isolation forest ignores temporal correlation.

## Incorporating LLMs into Data Streams
LLMs have the ability to explain issues in natural language, but are poor at processing large quantities of data. Marking anomalies in advance to reduce the workload and then asking the algorithm to explain the anomaly in natural languages gives data scientists a starting point for tracing issues. 

## Getting Started
Download and install Ollama. Fill out the `.env` file. Make sure to select Ollama models with low parameter counts (e.g. lfm2.5-thinking:1.2b) 

## Ensuring consistent results
Ollama's structured output means that the results of each prompt will be sent in a consistent format between prompts.
