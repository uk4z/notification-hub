# Notification hub

This project will try to implement a feature transforming voice messages into written notifications. Here you will be able to understand each step of the process used to build this feature.

The project is divided into two main components: summarization and speech-to-text. First we will be handling summarization. 

## Text summarization

The goal is to summarize conversational texts into short summaries. We will be covering the two different types of summarization: extractive and abstractive. On the one hand, we extract the sentences of the text containing most of the meaning. On the other hand, we generate new sentences summarizing the text. We will be confronting both types through an example. 

We have written a random story which will have to be summarize: 

*Sarah: (sighs) Hey, Michael. I hope you're doing well. I know it's been a while since we caught up, but I really need to talk to someone, and you've always been such a great friend.*

*(softly) So, Francis and I...we broke up. It happened a few days ago, and I'm struggling to cope with it. It's been really hard on me. You know how close we were, and this just feels like a huge void in my life right now.*

*I remember how you were there for me when we first got together, and now, I don't know who else to turn to. I miss your comforting words, your wisdom, and your laughter. I just need my friend right now, Michael.*

*(voice cracking) I was wondering if we could meet up soon, maybe this weekend? I'd really appreciate your support and advice. You've always had a way of helping me see things more clearly, and I really need that right now.*

*I hope you're not too busy, but if you are, I completely understand. Just hearing your voice in person would mean the world to me. I could use a good friend by my side, especially right now. Let me know if you're available, and we can plan something. Thanks, Michael. I miss you and can't wait to see you.*


### Extractive summarization

Code can be find on `extractive_sum.py`.

We will use a simple approach to handle this type of summarization. The idea is to filter words carrying the context and observe the most used ones. For example, link words like *and, for, or, so, therefore, ...* ar not really suitable as they do not really help define the context. We assume that most of the text meaning is contained in a few words. In the end, the summary will be composed of few sentences containing most of the context. In order to do so we will be weighting each sentence accordingly.

We will have to initialize our variables and do those imports:  

```
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

nlp = spacy.load("en_core_web_md")
text = open_file()
doc = nlp(text)
```

This will be helpful to process each raw sentences first by filtering the ponctuation and then keep only meaningful words using **Spacy tools**. We will rely only on english imputs for this type of summarization.

```ruby
def most_used_words(doc, n_words): 
    """
    Input: 
        doc: nlp item containing the text
        n_words: the nth first most used words
    Return:
        output: List[(word: string, frequency: int)] (sorted)
    """
    if not doc: 
        raise TypeError("The doc is null.")

    if n_words <= 0:
        raise ValueError('Please provide a valid number of words (strictly positive)')
    
    keyword = {}
    stopwords = list(STOP_WORDS)
    pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

    for token in doc: 
        if (token.text in stopwords or token.text in punctuation):
            continue 
        if (token.pos_ in pos_tag):
            keyword[token.text] = keyword.get(token.text, 0) + 1

    return sorted(keyword.items(), key=lambda tuple: tuple[1], reverse=True)[:n_words]
```

 This function basically counts the number of word apparitions. However, for more readability we are going to normalize those by the count of the most used word. 

 ```ruby
def normalization(words):
    """
    Input:
        words: List[(word: string, frequency: int)]
    Return: 
        output: Dic[(word: string | normalized_frequency: float)]
    """
    if words is None or words == []:
        raise ValueError("Please provid a non empty list")
    
    max_freq = words[0][1]
    weighted_words = {}

    for (word, freq) in words: 
        weighted_words[word] =  freq/max_freq
    
    return weighted_words
 ``` 


 Now, we can weight each sentences with the normalized words. This function will do the work for us. 

```ruby
 def weighted_sentences(doc, weighted_words, n_sents):
    """
    Input: 
        weighted_words: Dic[((word: string | weight: float))]
        doc:nlp item containing the text
        n_words: the nth first most used words
    Return:
        output: List[(sentence: spacy.token.span.Span, weight: float)] 

    """

    if not doc: 
        raise TypeError("No doc.")
    
    if not weighted_words: 
        raise ValueError("No weighted words.")
    
    if n_sents > len(list(doc)) : 
        message = "There is not enough words in the document. It has " + str(len(list(doc))) +" words but you want to get " + str(n_sents) + "."
        raise ValueError(message)

    sent_strength = {}

    for sent in doc.sents:
        sent_weight = 0
        for token in sent:
            sent_weight += weighted_words.get(token.text, 0.0)
        sent_strength[sent] = sent_weight
    
    return sorted(sent_strength.items(), key=lambda tuple: tuple[1], reverse=True)[:n_sents]
```

After processing the input we have been able to get the following summary: 

*I know it's been a while since we caught up, but I really need to talk to someone, and you've always been such a great friend.*
*I just need my friend right now, Michael.*
*You know how close we were, and this just feels like a huge void in my life right now.*

It is not that bad but we can easily find a more satisfying summary:

*So, Francis and I...we broke up.*
*I was wondering if we could meet up soon, maybe this weekend?*
*I'd really appreciate your support and advice.*

Here, we have an additional information: Sarah broke up with Francis. Unfortunately it might be the most important data to understand the reason of the audio message. But still as an extractive summary, it is lacking of smoothness in terms of readability. Therefore, we will not look further into this type of summary and try to get a better version with abstractive summarization. 


### Abstractive summarization

Code can be find on `abstractive_sum.py`.

This type of summarization is handled with artificial intelligence models. We will be using HuggingFace's library **Transformers**. This will comes handy to get models and datasets and create a fine-tuned model that will be fitting our purpose. Again, we will have to do some imports before starting to use AI models.

We will be working with *sshleifer/distilbart-cnn-12-6* which is a lighter version of meta's model *bart* which is a fine model to handle text summarization. In order to make conversationnal summaries, we will train it with the `samsum` dataset. It is organised as follow: 

| id | dialogue | summary |
| ------------- | ------------- | ------------- |
| 13818513 | Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-) | Amanda baked cookies and will bring Jerry some tomorrow. |
| 13728867 | Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great | Olivia and Olivier are voting for liberals in this election. |
| ... | ... | ... |

We will need to do some imports. Those will come in handy during the training. 

```
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
import numpy as np
from nltk.tokenize import sent_tokenize
import rouge_score
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
```

We can initialize our variables. 

```
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
dataset = load_dataset("samsum")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

The *tokenizer* is a useful tool to handle communications between humans and models. It can convert sentences readable by humans to list of integers readable by models or the contrary. 
The *model* is the core of the process. It will be trained to generate conversational summaries. 
The *dataset* contains the data to train properly the model. 
The *data_collator* is a useful tool to accelerate the training process. It will create batches of data so as to process several inputs at the same time. 

The first thing to do is to preprocess the data and tokenize the strings. The reference summaries will be tokenized in a new column called `labels`.

```ruby
def preprocess_dataset(examples):
    model_inputs = tokenizer(examples["dialogue"])
    labels = tokenizer(examples["summary"])
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_dataset = dataset.map(preprocess_dataset, batched=True)
```

The next thing to do is to set up the weights of the model. Here are the value we used. 

```
batch_size = 8 
num_train_epochs = 8 

logging_steps = len(tokenized_dataset["train"])
model_name = checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-summup",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    weight_decay=0.01, 
    save_total_limit=3, 
    num_train_epochs=num_train_epochs, 
    predict_with_generate=True,
)
```

Our approach is to do supervised learning, therefore we have to compute a metric in order to tell the model wether it is performant or not. To do so, we used a useful metric called **ROUGE** score (Recall-Oriented Understudy for Gisting Evaluation). The metrics compare our reference summary against the generated one and give us a f1-score telling us about the accuracy of the generation. 

```ruby 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # We filter the labels with the token -100 because it is the padding token value
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".joint(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}
```

Then we have everything to build our trainer and generate our fine-tune model. 

```ruby
trainer = Seq2SeqTrainer(
    model, 
    args, 
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

evaluation = trainer.evaluate()
```

Fine-tuning requires a lot of resources and it will take a really long time to train it on a single GPU. Some applications are available out there to train a model however HuggingFace's library provides a fine-tuned model that fits perfectly to our purpose so why don't we use it instead and save the computational cost. From now on, we will be using `philschmid/bart-large-cnn-samsum` model. 

```
from transformers import pipeline

pipe = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
```

Let's see how it will summarize our example.

*Sarah and Francis broke up a few days ago. Sarah needs Michael's support and advice. Sarah wants to meet up with Michael this weekend. Sarah will let Michael know if he's available. Sarah misses Michael and wants to talk to him. Sarah can't wait to see Michael.*

Here, we have all of the information we are looking for.  