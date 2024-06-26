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
from datasets import load_dataset
from evaluate import load
import evaluate
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator


cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'

model_small = "google/flan-t5-small"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_small)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)

# raw_datasets = load_dataset("xsum", cache_dir=cache_dir)
raw_datasets = load_dataset("glue", "wnli", cache_dir=cache_dir)
task = 'summarization'

# books = load_dataset("wmt14", "fr-en", split='test', cache_dir=cache_dir)
# metric = evaluate.load("sacrebleu", cache_dir=cache_dir)
# max_input_length = 1024
# max_target_length = 128

# source_lang = "en"
# target_lang = "fr"
# prefix = "translate English to French: "

max_input_length = 1024
max_target_length = 128
prefix = "natural language inference output entailment or not entailment: "

def preprocess_function(data_points):
    # inputs = [prefix + doc for doc in examples["document"]]
    inputs = [prefix + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
labels = tokenizer.batch_decode(tokenized_datasets["test"]["labels"], skip_special_tokens=True)
test_input_ids = tokenized_datasets["test"]["input_ids"]

for importance in [1e-2, 1e-4, 1e-0]:
    for epoch in [2]:
        model_small_ewc = f"/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_after_translation_importance_{'{:.0e}'.format(importance)}_epoch{epoch}"
        print(model_small_ewc)
        model_small_ewc = AutoModelForSeq2SeqLM.from_pretrained(model_small_ewc, local_files_only=True).to(device)
        results_small_ewc = {}
        progress_bar = tqdm(range(len(test_input_ids)))
        num_right = len(test_input_ids)
        group_len = 50
        for a in range(0, len(test_input_ids)//group_len):
            output_list, ref_list, src_list = [], [], []
            for b in range(group_len):
                index = a * group_len + b
                if index >= len(test_input_ids):
                    continue
                test_tensor = torch.tensor([test_input_ids[index]]).to(device)
                preds = model_small_ewc.generate(test_tensor, max_new_tokens=max_target_length, do_sample=False)                                   
                preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                output_list += preds
                ref_list.append(labels[index])
                # src_list.append(raw_datasets["test"][index]["document"])
                src_list.append(raw_datasets["test"][index]["dialogue"])
                
            data = convert_to_json(output_list=output_list, 
                                   src_list=src_list, ref_list=ref_list)
          
            evaluator = get_evaluator(task, cache_dir=cache_dir)
            
            eval_scores = evaluator.evaluate(data)
            # except ZeroDivisionError:
            #     continue
            # print(eval_scores)
            for eval_score in eval_scores:
                for key, value in eval_score.items():
                    if key not in results_small_ewc:
                        results_small_ewc[key] = value
                    else:
                        results_small_ewc[key] += value
            progress_bar.update(group_len)
        results_small_ewc_agg = {}
        
        for k, v in results_small_ewc.items():
            results_small_ewc_agg[k] = v/num_right
        print(f"For importance {importance} epoch {epoch}, the average score is: ")
        print(results_small_ewc_agg)
        print(f"Number of non empty answers is {num_right}")

# for importance in [1e-2, 1e-0]:
#     for epoch in [2, 1, 0]:
    
#         model_small_ewc = f"/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_importance_{'{:.0e}'.format(importance)}_epoch{epoch}"
#         model_small_ewc = AutoModelForSeq2SeqLM.from_pretrained(model_small_ewc, local_files_only=True).to(device)
#         scores_ewc = []
#         progress_bar = tqdm(range(len(books)))
#         for i in range(len(books)):
#             text = prefix + books[i]["translation"][source_lang]
#             ref = books[i]["translation"][target_lang]
#             inputs = tokenizer(text, return_tensors="pt").to(device).input_ids
#             preds_tokenized = model_small_ewc.generate(inputs, max_new_tokens=128, do_sample=False) 
#             preds = tokenizer.batch_decode(preds_tokenized)
            
#             bleu_score = metric.compute(predictions=preds, references=[ref])
#             scores_ewc.append(bleu_score["score"])
#             progress_bar.update(1)
#         print(f"For importance {importance}, epoch {epoch} the average score on the test set is {sum(scores_ewc)/len(books)}")
