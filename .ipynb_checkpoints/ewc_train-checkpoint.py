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
from ewc_utils import EWC, ewc_train, evaluate, flatten_params, recover_flattened
import pickle
from matplotlib import pyplot as plt


cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'
model_small = "google/flan-t5-small"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_small)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)

data_points = load_dataset("race", "all", cache_dir=cache_dir)
summary_datapoints = load_dataset("xsum", cache_dir=cache_dir)
data_points = data_points.filter(lambda x: len(x['options']) == 4)

# Preprocess functions
index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}
max_input_length = 1024
max_target_length = 128

def preprocess_function_summary(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_summary = summary_datapoints.map(preprocess_function_summary, batched=True)

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

tokenized_datasets = data_points.map(preprocess_function, batched=True)

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
with open('/scratches/dialfs/alta/hln35/distillation/QA_large_model_race_probability_output.txt') as f:
    large_model_outputs = json.load(f)
tokenized_datasets["train"] = tokenized_datasets["train"].add_column("labels", large_model_outputs)

# Load EWC object from file
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_translate_instance.txt", "rb") as fp:
    ewc_race_tran = pickle.load(fp)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_translate_10k_instance_use_ref.txt", "rb") as fp:
    ewc_race_tran_ref = pickle.load(fp)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_summary_instance.txt", "rb") as fp:
    ewc_race_sum = pickle.load(fp)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_summary_10k_instance_use_ref.txt", "rb") as fp:
    ewc_race_sum_ref = pickle.load(fp)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_data2text_1k_each_instance.txt", "rb") as fp:
    ewc_race_data2text = pickle.load(fp)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_title_generation_1k_each_instance.txt", "rb") as fp:
    ewc_race_title_sum = pickle.load(fp)

# flatten_sum_ewc_parameters = flatten_params(ewc_race_sum._precision_matrices)
# flatten_tran_ewc_parameters = flatten_params(ewc_race_tran._precision_matrices)

# sorted_params_sum, sorted_indices_sum = torch.sort(flatten_sum_ewc_parameters["params"], dim=0 ,descending=True)
# sorted_params_tran, sorted_indices_tran = torch.sort(flatten_tran_ewc_parameters["params"], dim=0 ,descending=True)

# top_index = []
# similarity_list = []
# set_top_sum, set_top_tran = set(), set()
# top_n = 30000
# for i in range(top_n):
#     set_top_sum.add(sorted_indices_sum[i][0].item())
#     # print(sorted_indices_sum[i][0].item())
#     set_top_tran.add(sorted_indices_tran[i][0].item())
#     # print(sorted_indices_tran[i][0].item())
    
#     if sorted_indices_sum[i][0].item() in set_top_tran:
#         top_index.append(i)
#     if sorted_indices_tran[i][0].item() in set_top_sum and sorted_indices_tran[i][0].item() != sorted_indices_sum[i][0].item():
#         top_index.append(i)
# for _, index in enumerate(top_index):
#     flatten_tran_ewc_parameters["params"][index] = float("inf")
# print(flatten_tran_ewc_parameters)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# model_tran_recovered = recover_flattened(flatten_tran_ewc_parameters["params"], flatten_tran_ewc_parameters["indices"], model.state_dict())
# ewc_race_tran._precision_matrices = model_tran_recovered
model_name = "/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_after_summarisation_ref_importance"
    
tokenized_datasets["train"].set_format("torch")
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]
eval_dataset = tokenized_datasets["validation"]
train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

tokenized_summary["train"].set_format("torch")
train_summary_set = tokenized_summary["train"].select(range(10000))
train_summary_dataloader = DataLoader(train_summary_set, batch_size=1)

validation_input_ids = tokenized_datasets["validation"]["input_ids"]
validation_labels = tokenized_datasets["validation"]['answer']

for importance in [1e-0, 1e-2, 1e-4]:
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    model_trained, loss_array, acc_array, eval_results = ewc_train(model=model, optimizer=optimizer, data_loader=train_dataloader, ewc=ewc_race_sum_ref, importance=importance, epochs=3, validation_input_ids=validation_input_ids, validation_labels=validation_labels, model_name=model_name)
        
