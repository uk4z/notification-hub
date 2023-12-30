# Notification hub

This project tries to implement partially a feature transforming voice messages into written notifications. Here you will be able to understand each step of the process used to build this feature. More precisely, it tackles the text summarization, one of the main NLP problem.

The goal is to summarize conversational texts into short summaries. We will be covering abstractive summarization using AI models.

## Project Management

The repository is organised as follow:

- The `main.ipybn` file contains all the code running properly.
- The `setup.ipybn` file is a frame of the `main` notebook. It contains code to setup the environment.
- The `statistic.ipybn` file is a frame of the `main` notebook. It contains code to analyse datasets and operate a statistical analysis.
- The `training.ipybn` file is a frame of the `main` notebook. It contains code to train the model and push it to HuggingFace.

The pipeline can be used directly on HuggingFace's inference API: https://huggingface.co/uk4zor/notification-hub.
Let's keep in mind that the model has been train on conversational dialogues. We advise to use the SAMSum dialogues to get the correct format. 

Or you can use the `transformer` library:

```ruby
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("summarization", model="uk4zor/notification-hub")
```



