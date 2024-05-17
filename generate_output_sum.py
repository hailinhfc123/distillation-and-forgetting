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
from ewc_utils import preprocess_function_summary

cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'

# cache_dir = "./.cache"
# os.environ['TRANSFORMERS_CACHE'] = './.cache'

model_small = "google/flan-t5-small"
model_large = "google/flan-t5-large"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
max_input_length = 1024
max_target_length = 128
tokenizer = AutoTokenizer.from_pretrained(model_large, cache_dir=cache_dir)

def output_generator(model, train_summary_set):
    progress_bar = tqdm(range(len(train_summary_set["input_ids"])))
    for i in range(len(train_summary_set["input_ids"])):
            test_tensor = torch.unsqueeze(train_summary_set["input_ids"][i],0).to(device)
            preds = model.generate(test_tensor, max_new_tokens=max_target_length, do_sample=False)[0]    
            #preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
            #print(preds_text)
            progress_bar.update(1)
            yield {"label_ids":preds}

def batch_generator(model, data_loader, batch_size):
    len_dataset = len(data_loader) * batch_size
    progress_bar = tqdm(range(len_dataset))
    return_pred_list = []
    for step, batch in enumerate(data_loader):
        test_tensor = batch["input_ids"].to(device)
        preds = model.generate(test_tensor, max_new_tokens=max_target_length, do_sample=False)   
        progress_bar.update(batch_size)
        # preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # print(preds_text)
        return_pred_list += preds
    return return_pred_list        


def generate_summary_outputs(model_name, dataset, path, batch_size, num_rows=1000):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    summary_datapoints = load_dataset(dataset, cache_dir=cache_dir)
    tokenized_summary = summary_datapoints.map(lambda b: preprocess_function_summary(b, max_input_length, max_target_length), 
                                               batched=True)
    tokenized_summary["train"].set_format("torch")
    if num_rows == "full":
        train_summary_set = tokenized_summary["train"]
    else:
        train_summary_set = tokenized_summary["train"].select(range(num_rows))

    dataset_loader = DataLoader(train_summary_set, batch_size=batch_size)
    return_dict = {"label_ids": batch_generator(model, dataset_loader, batch_size=batch_size)}
    dataset = Dataset.from_dict(return_dict)
    # dataset = Dataset.from_generator(lambda : output_generator(model, train_summary_set))
    # max_tokens_output_len = max(dataset["label_ids"], key=len)
    # dataset = dataset.map()
    torch.save(dataset, path)
    return dataset

dataset = generate_summary_outputs(model_large, "EdinburghNLP/xsum", "./output_xsum_from_t5_large_full.pt", batch_size=100, num_rows="full")

            
        
        
