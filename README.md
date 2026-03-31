# Multimodal Anomaly Detection with LLMs
By combining a relatively simple anomaly detection method, such as a Markov chain, with a lightweight large language model, we can create an anomaly detection system which helps reduce the odds of a false positive. Multimodal detection also reduces computational overhead as the data is reduced by the more rapid first detection pass. A multimodal system works well with live deployments and continuous data streams.

## Getting Started
Download and install Ollama. Fill out the `.env` file. Make sure to select Ollama models with low parameter counts (e.g. lfm2.5-thinking:1.2b) 

## Ensuring consistent results
Ollama's structured output means that the results of each prompt will be sent in a consistent format between prompts.
