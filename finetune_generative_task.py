import os
from torch.optim import AdamW
import torch
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, AutoModel, AutoModelForSeq2SeqLM
from datasets import load_dataset
import json, re
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from ewc_utils import EWC, ewc_train, evaluate, flatten_params, recover_flattened, normal_train, pad_dataset
import pickle
from matplotlib import pyplot as plt
from transformers import DataCollatorWithPadding


cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'
model_small = "google/flan-t5-small"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_small)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)

# Preprocess functions
index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}
max_input_length = 1024
max_target_length = 128

def preprocess_function_summary(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_function(data_points):
    prefix = "answer this question by choosing the best choice either A, B, C, or D. Given the context is:"
    inputs = []
    for i in range(len(data_points["question"])):
        if len(data_points["options"][i]) != 4:
            continue
        labels = list(ans_to_index.keys())
        q = data_points["question"][i]
        choices = ""
        choice = ""
        for t in range(len(labels)):
            choices += labels[t] + " " + data_points["options"][i][t] + ". "
            
        text = prefix + data_points["article"][i] + q + ". Choices: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    
    return model_inputs


def compute_loss_generate(input_ids, max_new_tokens, model, tokenizer, device):
    
    decoder_input_ids = tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"
    
    # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
    outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    
    # get encoded sequence
    encoded_sequence = (outputs.encoder_last_hidden_state,)
    # get logits
    lm_logits = outputs.logits
    # print(lm_logits)
    # sample last token with highest prob
    next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    l = torch.max(lm_logits[:, -1:])
    
    # concat
    decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
    next_decoder_input_ids = "0"
    no_tokens = 1
    while next_decoder_input_ids and next_decoder_input_ids != 1 and no_tokens<=max_new_tokens:
        lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
        l = torch.add(l,torch.max(lm_logits[:, -1:]))
        # sample last token with highest prob again
        next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
        # concat again
        decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
        no_tokens += 1
    l.backward()
    
# Load labels from large model results
# tokenized_datasets["train"] = tokenized_datasets["train"].add_column("labels", large_model_outputs)
print("loading the output")
# large_model_outputs = torch.load("/scratches/dialfs/alta/hln35/output_xsum_from_t5_large_50k.pt")
large_model_outputs = torch.load("/scratches/dialfs/alta/hln35/output_xsum_from_t5_large_full.pt")
print("output is loaded")
#max_tokens_output_len = max(len(large_model_outputs["label_ids"][i]) for i in range(len(large_model_outputs["label_ids"])))  
max_tokens_output_len = max_target_length
print("Found max token output length") 
large_model_outputs = large_model_outputs.map(lambda b : pad_dataset(b, tokenizer, max_tokens_output_len))

batch_size = 4
# model_name = f"/scratches/dialfs/alta/hln35/model/flant5_small_distill_xsum_batchsize_{batch_size}_10k_samples"
summary_datapoints = load_dataset("xsum", cache_dir=cache_dir)
tokenized_summary = summary_datapoints.map(preprocess_function_summary, batched=True)
tokenized_summary["train"].set_format("torch")
train_summary_set = tokenized_summary["train"]
# train_summary_set = tokenized_summary["train"].select(range(200))
train_summary_set = train_summary_set.remove_columns("labels")
train_summary_set = train_summary_set.add_column("labels", large_model_outputs["label_ids"])
train_summary_dataloader = DataLoader(train_summary_set, batch_size=batch_size)

model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

print("Start training")    
# model_trained = normal_train(model=model, optimizer=optimizer, data_loader=train_summary_dataloader, epochs=3, comment_to_file_name=f"flant5_small_finetune_xsum_batchsize_{batch_size}_10k_samples", batch_size=batch_size, evaluator=None)
model_trained = normal_train(model=model, optimizer=optimizer, data_loader=train_summary_dataloader, epochs=3, comment_to_file_name=f"flant5_small_distill_xsum_batchsize_{batch_size}_full_samples_with_attention_mask", batch_size=batch_size, evaluator=None)
        
