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
from ewc_utils import L2, ewc_train, evaluate, flatten_params, recover_flattened, preprocess_function_summary, pad_dataset, preprocess_function_race
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

race_datapoints = load_dataset("race", "all", cache_dir=cache_dir)
summary_datapoints = load_dataset("xsum", cache_dir=cache_dir)
race_datapoints = race_datapoints.filter(lambda x: len(x['options']) == 4)

# Preprocess functions
index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}
max_input_length = 1024
max_target_length = 128


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
    
batch_size = 4
file_name = f"flant5_small_l2_train_finetune_race_keep_xsum_batchsize_{batch_size}_full_samples"

tokenized_race = race_datapoints.map(lambda b: preprocess_function_race(b, max_input_length), 
                                               batched=True)
tokenized_summary = summary_datapoints.map(lambda b: preprocess_function_summary(b, max_input_length, max_target_length), 
                                               batched=True)
l2 = L2(model=model)

large_model_outputs = torch.load("/scratches/dialfs/alta/hln35/output_xsum_from_t5_large_50k.pt")
max_tokens_output_len = max_target_length
large_model_outputs = large_model_outputs.map(lambda b : pad_dataset(b, tokenizer, max_tokens_output_len))

tokenized_race["train"].set_format("torch")
train_race_dataset = tokenized_race["train"]
test_race_dataset = tokenized_race["test"]
eval_race_dataset = tokenized_race["validation"]
train_race_dataloader = DataLoader(train_race_dataset, batch_size=1)
test_race_dataloader = DataLoader(test_race_dataset, batch_size=1)
eval_race_dataloader = DataLoader(eval_race_dataset, batch_size=1)

tokenized_summary["train"].set_format("torch")
train_summary_set = tokenized_summary["train"]
# train_summary_set = train_summary_set.remove_columns("labels")
# train_summary_set = train_summary_set.add_column("labels", large_model_outputs["label_ids"])
train_summary_dataloader = DataLoader(train_summary_set, batch_size=batch_size)

validation_input_ids = eval_race_dataset["input_ids"]
validation_labels = eval_race_dataset['answer']

# for importance in [10, 1e-0, 5e-1, 1e-1 ,1e-2, 1e-4]:
for importance in [1e-2, 1e-4]:
# for importance in [ 1e-0, 1e+2]:
# for importance in [ 1e+4, 1e+6]:
# for importance in [10]:
# for importance in [ 1e-0]:
# for importance in [ 5e-1]:
# for importance in [ 1e-1]:
# for importance in [ 1e-2]:
# for importance in [ 1e-4]:
    print(importance)
    comment_to_file_name = file_name + f"_importance_{'{:.0e}'.format(importance)}"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # model_trained = ewc_train(model=model, optimizer=optimizer, data_loader=train_summary_dataloader, ewc=ewc_race, importance=importance, epochs=3, batch_size=batch_size, validation_input_ids=validation_input_ids, validation_labels=validation_labels, comment_to_file_name=comment_to_file_name)
    model_trained = ewc_train(model=model, optimizer=optimizer, data_loader=train_race_dataloader, ewc=l2, importance=importance, epochs=3, batch_size=batch_size, validation_input_ids=validation_input_ids, validation_labels=validation_labels, comment_to_file_name=comment_to_file_name)
        
