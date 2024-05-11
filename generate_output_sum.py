import os
from torch.optim import AdamW
import torch
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, AutoModel, AutoModelForSeq2SeqLM
import json, re
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from evaluate import load
import json

cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'

# cache_dir = "./.cache"
# os.environ['TRANSFORMERS_CACHE'] = './.cache'

model_small = "google/flan-t5-small"
model_large = "google/flan-t5-large"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_input_length = 1024
max_target_length = 128
tokenizer = AutoTokenizer.from_pretrained(model_large, cache_dir=cache_dir)

def preprocess_function_summary(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def output_generator(model, train_summary_set):
    progress_bar = tqdm(range(len(train_summary_set["input_ids"])))
    for i in range(len(train_summary_set["input_ids"])):
            test_tensor = torch.unsqueeze(train_summary_set["input_ids"][i],0).to(device)
            preds = model.generate(test_tensor, max_new_tokens=max_target_length, do_sample=False)                                   
            preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
            progress_bar.update(1)
            yield {"label_ids":preds}


def generate_summary_outputs(model_name, dataset, path, num_rows=1000):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    summary_datapoints = load_dataset(dataset, cache_dir=cache_dir)
    tokenized_summary = summary_datapoints.map(preprocess_function_summary, batched=True)
    tokenized_summary["train"].set_format("torch")
    train_summary_set = tokenized_summary["train"].select(range(num_rows))
    dataset = Dataset.from_generator(lambda : output_generator(model, train_summary_set))
    torch.save(dataset, path)
    return dataset

dataset = generate_summary_outputs(model_large, "xsum", "./output_xsum_from_t5_large.pt", num_rows=10000)

            
        
        