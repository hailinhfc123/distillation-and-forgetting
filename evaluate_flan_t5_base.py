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
import evaluate
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from rouge import Rouge
from ewc_utils import exact_match, em_evaluator, normalize_answer, evaluate
import json



cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'

model_small = "google/flan-t5-small"
model_base = "google/flan-t5-base"

if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_base, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_base, cache_dir=cache_dir).to(device)

index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}

# raw_datasets = load_dataset("xsum", cache_dir=cache_dir)
# raw_datasets = load_dataset("samsum", cache_dir=cache_dir)


# books = load_dataset("wmt14", "fr-en", split='test', cache_dir=cache_dir)
# metric = evaluate.load("sacrebleu", cache_dir=cache_dir)
# max_input_length = 1024
# max_target_length = 128

# source_lang = "en"
# target_lang = "fr"
# prefix = "translate English to French: "

max_input_length = 1024
max_target_length = 128


def evalute_summary(model_small_ewc, tokenizer, test_input_ids, labels):
    task = 'summarization'
    results_small_ewc = {}
    progress_bar = tqdm(range(len(test_input_ids)))
    num_right = len(test_input_ids)
    group_len = 20
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
#        print(f"For model {model}, the average score is: ")
    print(results_small_ewc_agg)
    print(f"Number of non empty answers is {num_right}")

def evalute_word_ouput(model_small_ewc, tokenizer, test_input_ids, labels, decode_output_dict, use_sentiment=True):
    results_small_ewc = {}
    scores = 0
    progress_bar = tqdm(range(len(test_input_ids)))
    num_right = len(test_input_ids)
    group_len = 1
    print(f"Test set length {len(test_input_ids)}")
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
        # print(labels)
#        print(output_list, ref_list, src_list)
        for i in range(len(ref_list)):
            if use_sentiment == False:
                if output_list[i] == ref_list[i]:
                    scores += 1
                if output_list[i] == "":
                    num_right -= 1
            else:
                if output_list[i] == "" or output_list[i].lower() not in decode_output_dict:
                    num_right -= 1
                elif decode_output_dict[output_list[i].lower()] == ref_list[i]:
                    # print(output_list[i].lower())
                    scores += 1
        
        progress_bar.update(group_len)
    
#        print(f"For model {model}, the average score is: ")
    print(scores)
    print(f"Number of non empty answers is {num_right}")

def evaluate_sentence_output(model_small_ewc, tokenizer, test_input_ids, labels, evaluator=Rouge()):
    results_small_ewc = {}
    progress_bar = tqdm(range(len(test_input_ids)))
    num_right = len(test_input_ids)
    group_len = 1
    print(f"Test set length {len(test_input_ids)}")
    
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
            ref_list += labels[index]
            # src_list.append(raw_datasets["test"][index]["document"])
            
        # data = convert_to_json(output_list=output_list, cores=src_list, ref_list=ref_list)
      
        # evaluator = get_evaluator(task, cache_dir=cache_dir)
        
        if len(output_list) != len(ref_list):
            output_list = output_list*(len(ref_list)//len(output_list))
        #print(output_list)
        #print(ref_list)
        if isinstance(evaluator, Rouge):
            if normalize_answer(output_list[0]) == "":
                eval_scores = results_small_ewc
                for rouge_metric, eval_score in eval_scores.items():
                    for key, value in eval_score.items():
                        eval_scores[rouge_metric][key] = 0
                        
            else:
                eval_scores = evaluator.get_scores(output_list, ref_list)[0]
        
            for rouge_metric, eval_score in eval_scores.items():
                for key, value in eval_score.items():
                    if rouge_metric not in results_small_ewc:
                        results_small_ewc[rouge_metric] = eval_score
                    else:
                        results_small_ewc[rouge_metric][key] += value
        else:
            eval_scores = evaluator(output_list, ref_list)
            if results_small_ewc:
                results_small_ewc["em"] += eval_scores
            else:
                results_small_ewc["em"] = eval_scores
            
        progress_bar.update(group_len)
    
    results_small_ewc_agg = {}
    if isinstance(evaluator, Rouge):
    
        for rouge_metric, eval_score in results_small_ewc.items():
            results_small_ewc_agg[rouge_metric] = {}
            
            for key, value in eval_score.items():
                results_small_ewc_agg[rouge_metric][key] = value/num_right
    else:
                results_small_ewc_agg["em"] = results_small_ewc["em"]/num_right
        
    
    print(results_small_ewc_agg)
    print(f"Number of non empty answers is {num_right}")

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
    
def preprocess_anli(data_points):
    prefix = 'Paragraph: '
    choices = 'Yes \n Not possible \n No'
    
    inputs = []
    for i in range(len(data_points["hypothesis"])):
        
        
        q = data_points["hypothesis"][i]
            
        text = prefix + data_points["premise"][i] + f'Based on the paragraph above can we conclude that "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs
    
def preprocess_sst2(data_points):
    prefix = 'Sentiment Analysis, paragraph: '
    choices = 'Negative \n Positive'
    
    inputs = []
    for i in range(len(data_points["sentence"])):
        
        
        # q = data_points["hypothesis"][i]
            
        text = prefix + data_points["sentence"][i] + f'Based on the paragraph above can we conlude that is positive or negative' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs

def preprocess_boolq(data_points):
    prefix = 'Paragraph: '
    choices = 'True \n False'
    
    inputs = []
    for i in range(len(data_points["passage"])):
        
        
        q = data_points["question"][i]
            
        text = prefix + data_points["passage"][i] + f'Based on the paragraph above: "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    model_inputs["label"] = data_points["answer"]
    
    
    return model_inputs

def preprocess_mnli(data_points):
    prefix = 'Paragraph: '
    choices = 'Neutral \n Contradiction \n Entailment'
    
    inputs = []
    for i in range(len(data_points["text1"])):
        
        
        q = data_points["text2"][i]
            
        text = prefix + data_points["text1"][i] + f'Based on the paragraph above can we conclude that "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs

def preprocess_piqa(data_points):
    prefix = 'Paragraph: '
    choices = 'Neutral \n Contradiction \n Entailment'
    
    inputs = []
    for i in range(len(data_points["text1"])):
        
        
        q = data_points["text2"][i]
            
        text = prefix + data_points["text1"][i] + f'Based on the paragraph above can we conclude that "{q}"' + ". Options: " + choices
        inputs.append(text)
    model_inputs = tokenizer(inputs, truncation=True)
    # model_inputs["labels"] = model_inputs["label"]
    
    
    return model_inputs

def preprocess_function_translate(examples):
    source_lang = "en"
    target_lang = "fr"
    prefix_translate = "translate English to French: "
    inputs = [prefix_translate + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    # model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    
    return model_inputs


def preprocess_function_summary(examples):
    prefix = "summarize: "
    # inputs = [prefix + doc for doc in examples["document"]]
    inputs = [prefix + doc for doc in examples["dialogue"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_nat_inst(examples):
    inputs = [prefix + doc for doc in examples["input"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    return model_inputs

def process_data_nat_inst(task_name):
    task_length = 100 
    with open(task_name, "r") as read_content: 
        fields = json.load(read_content)
        fields.pop("Instances")
    dataset = load_dataset('json', data_files=task_name, field="Instances")
    dataset_formatted = Dataset.from_dict(dataset["train"][:task_length])
    fields["Instances"] = dataset_formatted
    return fields
    
decode_output_dict_anli = {"yes":0, "0":0, "1":1, "2":2, "it's impossible to say":1, "no":2}
decode_output_dict_sst2 = {"negative":0, "positive":1, "0":0, "1":1}
decode_output_dict_mnli = {"neutral":0, "contradiction":1, "entailment":2, "0":0, "1":1, "2":2, "it's impossible to say":1, "no":2, "yes":0}
decode_output_dict_boolq = {"true":True, "false":False, "yes": True, "no": False}


task_list = [("facebook/anli", "test_r1", preprocess_anli, decode_output_dict_anli), ("sst2", "validation", preprocess_sst2, decode_output_dict_sst2), ("google/boolq", "validation", preprocess_boolq, decode_output_dict_boolq), ("SetFit/mnli", "validation", preprocess_mnli, decode_output_dict_mnli), ("facebook/anli", "test_r2", preprocess_anli, decode_output_dict_anli), ("facebook/anli", "test_r3", preprocess_anli, decode_output_dict_anli)]
# ("piqa", "validation", preprocess_anli, decode_output_dict_anli),

with open("/scratches/dialfs/alta/hln35/natural-instructions/splits/default/test_tasks.txt", "r") as file: 
    task_list_nat_inst = file.read().split("\n")
    task_list_nat_inst = list(map( lambda x: "/scratches/dialfs/alta/hln35/natural-instructions/tasks/" + x + ".json" if x else None, task_list_nat_inst))

def task_list_filtered_category(task_list, category):
    res = []
    for task_name in task_list:
        if task_name:
            with open(task_name, "r") as read_content: 
                fields = json.load(read_content)
                # print(fields["Categories"])
                if category in fields["Categories"]:
                    res.append(task_name)
    return res

tasks_data_to_text = task_list_filtered_category(task_list_nat_inst, "Data to Text")
tasks_title_generation = task_list_filtered_category(task_list_nat_inst, "Title Generation")

print(len(tasks_data_to_text))
print(len(tasks_title_generation))

# race_data_points = load_dataset("race", "all", cache_dir=cache_dir)
# race_data_points = race_data_points.filter(lambda x: len(x['options']) == 4)
# tokenized_race = race_data_points.map(preprocess_function_race, batched=True)
# test_race_set = tokenized_race["test"]
# test_input_ids = test_race_set["input_ids"]
# test_labels = test_race_set['answer']

# model_list = ["/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_importance_1e+00_epoch2",
#              "/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_importance_1e-02_epoch2",
#               "/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_importance_1e+00_epoch1",
#              "/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_importance_1e-02_epoch1",
#              "/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_after_translation_importance_1e-04_epoch1", 
#              model_base]

# for m in model_list:
#     print(f"We are evaluating model {m}")
#     model = AutoModelForSeq2SeqLM.from_pretrained(m).to(device)
#     model_test_accuracy = evaluate(model, test_input_ids, test_labels)
#     print(f"Accuracy is {model_test_accuracy/len(test_input_ids)}")
    

            
        


