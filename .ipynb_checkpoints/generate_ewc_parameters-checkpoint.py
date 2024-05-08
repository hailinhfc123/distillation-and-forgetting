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
from ewc_utils import EWC, ewc_train, evaluate, compute_loss_generate
from ewc_utils import process_data_nat_inst
import pickle
from matplotlib import pyplot as plt


cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'
model_small = "google/flan-t5-small"
model_base = "google/flan-t5-base"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_small)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)


source_lang = "en"
target_lang = "fr"
prefix_translate = "translate English to French: "
index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}
max_input_length = 800
max_target_length = 128

def preprocess_function_nat_inst(examples):
    inputs = [prefix + doc for doc in examples["input"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    return model_inputs

def preprocess_anli(data_points):
    prefix = 'Paragraph: '
    choices = 'A. Yes \n B. Not possible \n C. No'
    
    inputs = []
    for i in range(len(data_points["hypothesis"])):
        
        
        q = data_points["hypothesis"][i]
            
        text = prefix + data_points["premise"][i] + f'. Answer this question by choosing the best choice either A, B, C. Based on the paragraph above can we conclude that "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs

def preprocess_sst2(data_points):
    prefix = 'Sentiment Analysis, paragraph: '
    choices = 'A. Negative \n B. Positive'
    
    inputs = []
    for i in range(len(data_points["sentence"])):
        
        
        # q = data_points["hypothesis"][i]
            
        text = prefix + data_points["sentence"][i] + f'. Answer this question by choosing the best choice either A, B. Based on the paragraph above can we conlude that is positive or negative' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs

def preprocess_boolq(data_points):
    prefix = 'Paragraph: '
    choices = 'A. False \n B. True'
    
    inputs = []
    for i in range(len(data_points["passage"])):
        
        
        q = data_points["question"][i]
            
        text = prefix + data_points["passage"][i] + f'. Answer this question by choosing the best choice either A, B. Based on the paragraph above: "{q}"' + ". Options: " + choices
        inputs.append(text)
        model_inputs = tokenizer(inputs, truncation=True)
        model_inputs["label"] = list(map(lambda x: 1 if x else 0, data_points["answer"]))
    
    
    return model_inputs

def preprocess_mnli(data_points):
    prefix = 'Paragraph: '
    choices = 'A. Entailment \n B. Neutral \n C. Contradiction'
    inputs = []
    for i in range(len(data_points["text1"])):
        
        
        q = data_points["text2"][i]
            
        text = prefix + data_points["text1"][i] + f'. Answer this question by choosing the best choice either A, B, C. Based on the paragraph above can we conclude that "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs


def preprocess_function_translate(examples):
    inputs = [prefix_translate + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    # model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_function_summary(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_race(data_points):
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
    
# translate_datapoints = load_dataset("wmt14", "fr-en", cache_dir=cache_dir)
# tokenized_translate = translate_datapoints.map(preprocess_function_translate, batched=True)
# # test_tensor = torch.tensor([tokenized_translate["train"][10]["input_ids"]]).to(device)
# # compute_loss_generate(test_tensor, max_target_length, model, tokenizer, device)
# # print(compute_loss_generate)
# tokenized_translate["train"].set_format("torch")
# # train_translate_set = tokenized_translate["train"]
# train_translate_set = tokenized_translate["train"].select(range(10000))
# # train_translate_set = train_translate_set.add_column("labels", torch.zeros((1,10000), device=device))
# train_translate_dataloader = DataLoader(train_translate_set, batch_size=1)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# ewc_race = EWC(model, train_translate_dataloader, use_generate=True, use_ref=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_translate_10k_instance_use_ref.txt", "wb") as fp:
#     pickle.dump(ewc_race, fp)


# summary_datapoints = load_dataset("xsum", cache_dir=cache_dir)
# tokenized_summary = summary_datapoints.map(preprocess_function_summary, batched=True)
# tokenized_summary["train"].set_format("torch")
# # train_summary_set = tokenized_summary["train"]
# train_summary_set = tokenized_summary["train"].select(range(10000))
# train_summary_dataloader = DataLoader(train_summary_set, batch_size=1)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
# ewc_race = EWC(model, train_summary_dataloader, use_generate=True, use_ref=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_base_after_summary_10k_instance_use_ref.txt", "wb") as fp:
#    pickle.dump(ewc_race, fp)

# # with open('/scratches/dialfs/alta/hln35/distillation/QA_large_model_race_probability_output.txt') as f:
# #     large_model_outputs = json.load(f)
# race_data_points = load_dataset("race", "all", cache_dir=cache_dir)
# race_data_points = race_data_points.filter(lambda x: len(x['options']) == 4)
# tokenized_race = race_data_points.map(preprocess_function_race, batched=True)
# train_labels_race = []
# tokenized_race["train"].set_format("torch")
# train_race_set = tokenized_race["train"].select(range(10000))
# for i in range(len(train_race_set)):
#     ans = tokenized_race["train"]["answer"][i]
#     ans_probability_distribution = [0.0 if t != int(ans_to_index[ans]) else 1.0 for t in range(4)]
#     train_labels_race.append(ans_probability_distribution)
# train_race_set = train_race_set.add_column("labels", train_labels_race)
# train_race_dataloader = DataLoader(train_race_set, batch_size=1)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
# ewc_race = EWC(model, train_race_dataloader, use_generate=False, use_ref=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_race_10k.txt", "wb") as fp:
#     pickle.dump(ewc_race, fp)

# anli_data_points = load_dataset("facebook/anli", split="train_r1", cache_dir=cache_dir)
# tokenized_anli = anli_data_points.map(preprocess_anli, batched=True)
# train_labels_anli = []
# for i in range(len(tokenized_anli)):
#     ans = tokenized_anli["label"][i]
#     ans_probability_distribution = [0.0 if t != int(ans) else 1.0 for t in range(3)]
#     train_labels_anli.append(ans_probability_distribution)
# tokenized_anli = tokenized_anli.add_column("labels", train_labels_anli)
# tokenized_anli.set_format("torch")
# train_anli_set = tokenized_anli
# # train_anli_set = tokenized_anli.select(range(10000))
# train_anli_dataloader = DataLoader(train_anli_set, batch_size=1)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
# ewc_anli = EWC(model, train_anli_dataloader, use_generate=False, use_ref=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_anli_train_r1.txt", "wb") as fp:
#     pickle.dump(ewc_anli, fp)

# mnli_data_points = load_dataset("SetFit/mnli", split="train", cache_dir=cache_dir)
# tokenized_mnli = mnli_data_points.map(preprocess_mnli, batched=True)
# train_labels_mnli = []
# tokenized_mnli.set_format("torch")
# train_mnli_set = tokenized_mnli.select(range(10000))
# for i in range(len(train_mnli_set)):
#     ans = tokenized_mnli["label"][i]
#     ans_probability_distribution = [0.0 if t != int(ans) else 1.0 for t in range(3)]
#     train_labels_mnli.append(ans_probability_distribution)
# train_mnli_set = train_mnli_set.add_column("labels", train_labels_mnli)
# train_mnli_dataloader = DataLoader(train_mnli_set, batch_size=1)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
# ewc_mnli = EWC(model, train_mnli_dataloader, use_generate=False, use_ref=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_mnli_train_10k.txt", "wb") as fp:
#     pickle.dump(ewc_mnli, fp)



sst2_data_points = load_dataset("sst2", split="train", cache_dir=cache_dir)
tokenized_sst2 = sst2_data_points.map(preprocess_sst2, batched=True)
train_labels_sst2 = []
tokenized_sst2.set_format("torch")
train_sst2_set = tokenized_sst2.select(range(10000))
for i in range(len(train_sst2_set)):
    ans = tokenized_sst2["label"][i]
    ans_probability_distribution = [0.0 if t != int(ans) else 1.0 for t in range(2)]
    train_labels_sst2.append(ans_probability_distribution)
train_sst2_set = train_sst2_set.add_column("labels", train_labels_sst2)
train_sst2_dataloader = DataLoader(train_sst2_set, batch_size=1)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
ewc_sst2 = EWC(model, train_sst2_dataloader, use_generate=False, use_ref=True)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_sst2_train_10k.txt", "wb") as fp:
    pickle.dump(ewc_sst2, fp)

boolq_data_points = load_dataset("google/boolq", split="train", cache_dir=cache_dir)
tokenized_boolq = boolq_data_points.map(preprocess_boolq, batched=True)
train_labels_boolq = []
tokenized_boolq.set_format("torch")
for i in range(len(tokenized_boolq)):
    ans = tokenized_boolq["label"][i]
    ans_probability_distribution = [0.0 if t != int(ans) else 1.0 for t in range(2)]
    train_labels_boolq.append(ans_probability_distribution)
tokenized_boolq = tokenized_boolq.add_column("labels", train_labels_boolq)
train_boolq_set = tokenized_boolq
train_boolq_dataloader = DataLoader(train_boolq_set, batch_size=1)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_base).to(device)
ewc_boolq = EWC(model, train_boolq_dataloader, use_generate=False, use_ref=True)
with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_boolq_train_full.txt", "wb") as fp:
    pickle.dump(ewc_boolq, fp)

# with open("/scratches/dialfs/alta/hln35/natural-instructions/splits/default/test_tasks.txt", "r") as file: 
#     task_list_nat_inst = file.read().split("\n")
#     task_list_nat_inst = list(map( lambda x: "/scratches/dialfs/alta/hln35/natural-instructions/tasks/" + x + ".json" if x else None, task_list_nat_inst))

# def task_list_filtered_category(task_list, category):
#     res = []
#     for task_name in task_list:
#         if task_name:
#             with open(task_name, "r") as read_content: 
#                 fields = json.load(read_content)
#                 # print(fields["Categories"])
#                 if category in fields["Categories"]:
#                     res.append(task_name)
#     return res

# tasks_data_to_text = task_list_filtered_category(task_list_nat_inst, "Data to Text")
# tasks_title_generation = task_list_filtered_category(task_list_nat_inst, "Title Generation")

# train_title_sum_set = None
# for dataset in tasks_title_generation:
#     task_length = 1000
#     dataset_dict = process_data_nat_inst(dataset, task_length)
#     prefix = dataset_dict["Definition"][0][0]
#     raw_datasets = dataset_dict["Instances"]

#     tokenized_datasets = raw_datasets.map(preprocess_function_nat_inst, batched=True)
#     tokenized_datasets.set_format("torch")
#     if train_title_sum_set == None:
#         train_title_sum_set = tokenized_datasets
#     else:
#         train_title_sum_set = torch.utils.data.ConcatDataset([train_title_sum_set, tokenized_datasets])
#     labels = tokenized_datasets["output"]

#     test_input_ids = tokenized_datasets["input_ids"]

# train_title_sum_dataloader = DataLoader(train_title_sum_set, batch_size=1)
# print(len(train_title_sum_dataloader))    
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# ewc_title_sum = EWC(model, train_title_sum_dataloader, use_generate=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_title_generation_1k_each_instance.txt", "wb") as fp:
#    pickle.dump(ewc_title_sum, fp)

# train_data2text_set = None
# for dataset in tasks_data_to_text:
#     task_length = 1000
#     dataset_dict = process_data_nat_inst(dataset, task_length)
#     prefix = dataset_dict["Definition"][0][0]
#     raw_datasets = dataset_dict["Instances"]

#     tokenized_datasets = raw_datasets.map(preprocess_function_nat_inst, batched=True)
#     tokenized_datasets.set_format("torch")
#     if train_data2text_set == None:
#         train_data2text_set = tokenized_datasets
#     else:
#         train_data2text_set = torch.utils.data.ConcatDataset([train_data2text_set, tokenized_datasets])
#     labels = tokenized_datasets["output"]

#     test_input_ids = tokenized_datasets["input_ids"]

# train_data2text_dataloader = DataLoader(train_data2text_set, batch_size=1)
# print(len(train_data2text_dataloader))    
# model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
# ewc_data2text = EWC(model, train_data2text_dataloader, use_generate=True)
# with open("/scratches/dialfs/alta/hln35/distillation/ewc_after_data2text_1k_each_instance.txt", "wb") as fp:
#    pickle.dump(ewc_data2text, fp)
    
