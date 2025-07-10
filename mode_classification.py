import json,os
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

label2id = {'dialogue': 0, 'action': 1, 'non-action': 2}


def jsonl2csv(json_path, csv_path):
    with open(json_path, 'r') as f:
        data = {'target': [], 'label': [], 'idx': []}
        idx = 0
        for line in f:
            if line.strip():
                line = json.loads(line)
                # print(line)
                data['target'].append(line['target'])
                data['label'].append(label2id[line['target_label']])
                data['idx'].append(idx)
                idx += 1
        df = pd.DataFrame(data=data)
        df.to_csv(csv_path, index=False)


def preprocess_function(examples):
    return tokenizer(examples['target'], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')


def model_init(path=None):
    if path is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path,num_labels=num_labels)
    return model


num_labels = 3
metric_name = "f1"
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"temp/mode_classification_{model_name}",
    evaluation_strategy="epoch",
    #save_steps="epoch",
    learning_rate=2.327e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    seed=32
)

import numpy as np

metric = load_metric('f1')

data_dir = os.path.join(os.environ['PROJECT_DIR'],'dataset/pg_data_full2/{}/book_selected_pairs_cfg1/{}_{}.{}')
json_train_file = data_dir.format('train','train','27000','jsonl')
csv_train_file = data_dir.format('train','train','27000','csv')
json_valid_file = data_dir.format('valid','valid','1080','jsonl')
csv_valid_file = data_dir.format('valid','valid','1080','csv')

def main():
    jsonl2csv(json_train_file,csv_train_file)
    jsonl2csv(json_valid_file,csv_valid_file)
    dataset = load_dataset('csv', data_files={'train':csv_train_file,
                                              'validate':csv_valid_file})
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    trainer = Trainer(
        model_init(path=f"temp/mode_classification_{model_name}/checkpoint-8440").to('cuda:0'),
        args=args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validate'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    #trainer.train()
    eval_results=trainer.evaluate()
    print(f'evaluation: {eval_results}')

if __name__ == '__main__':
    main()