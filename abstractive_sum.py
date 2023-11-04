from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
import numpy as np
from nltk.tokenize import sent_tokenize
import rouge_score
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer


dataset = load_dataset("samsum")
'''I will use the samsum dataset with the distilbart model. See: https://huggingface.co/philschmid/distilbart-cnn-12-6-samsum'''

print(dataset)

def get_samples(dataset, num_samples=1, seed=42):
    samples = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    return samples

checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def preprocess_dataset(examples):
    model_inputs = tokenizer(examples["dialogue"])
    labels = tokenizer(examples["summary"])
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

samples = get_samples(dataset, num_samples=1)

tokenized_dataset = dataset.map(preprocess_dataset, batched=True)


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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != 100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".joint(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}


tokenized_dataset = tokenized_dataset.remove_columns(
    dataset["train"].column_names
)

features = [tokenized_dataset["train"][i] for i in range(2)]

print(data_collator(features))

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

print(evaluation)
