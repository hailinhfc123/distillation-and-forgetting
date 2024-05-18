from copy import deepcopy
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import numpy as np
import os
from torch.optim import AdamW
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, AutoModel, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
import json, re
import string
from torch.utils.tensorboard import SummaryWriter
from rouge import Rouge
import datetime
from UniEval.metric.evaluator import SumEvaluator
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return t
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
index_to_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
ans_to_index = {"A" : "0", "B" : "1", "C" : "2", "D": "3"}
ans_id_dict = {71: "A", 272: "B", 205: "C", 309: "D"}
id_ans_dict = {"A": 71, "B": 272, "C": 205, "D": 309}
model_small = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_small)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)
class EWC(object):
    def __init__(self, model: nn.Module, dataset:torch.utils.data.DataLoader, use_generate:bool, use_ref:bool):

        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.use_generate = use_generate
        self.use_ref = use_ref
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(device)

    def _diag_fisher(self, max_target_length=128):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(device)

        self.model.eval()
        for input in tqdm(self.dataset):
            self.model.zero_grad()
            # input = variable(input)
            if self.use_generate:
                input_ids = input["input_ids"].to(device)
                labels = input["labels"].to(device)
                # output = self.model.generate(input, max_new_tokens=max_target_length, return_dict_in_generate=True, output_scores=True)
                
                # transition_scores = model.compute_transition_scores(
                #     output.sequences, output.scores, normalize_logits=True
                # ).requires_grad_(True)
                # loss = torch.negative(torch.sum(transition_scores)).requires_grad_(True)
                # # loss = self.model(input_ids=input, labels=labels).loss
                # print(loss)
                
                # loss.backward()
                if self.use_ref:
                    compute_loss_generate(input_ids, max_target_length, self.model, tokenizer, device, labels)
                else:
                    compute_loss_generate(input_ids, max_target_length, self.model, tokenizer, device)
                
            else:             
                labels = input["labels"].clone().detach().requires_grad_(True).to(device)
                input = input["input_ids"].to(device)
                output = self.model(input, decoder_input_ids=torch.tensor([[self.model.config.decoder_start_token_id]]).to(device)).logits.view(1, -1)
                
                # output = self.model(input, decoder_input_ids=torch.tensor([[self.model.config.decoder_start_token_id]]).to(device)).logits
                
                labels = labels.squeeze()
                # preds_prob = []
                # len_labels_set = len(labels)
                # for t in ans_id_dict.keys():
                #     if len_labels_set == 0:
                #         continue
                #     preds_prob.append(output[0][t].item())
                #     len_labels_set -= 1
                # # labels.retain_grad()
                # print(preds_prob)
                
                # preds = output[0, 0, tuple(list(ans_id_dict.keys()))]
                # label = output.max(1)[1].view(-1)
                label = int(torch.argmax(labels))
                label_id = torch.tensor([id_ans_dict[index_to_ans[label]]]).to(device)
                # loss = torch.nn.CrossEntropyLoss()(labels, preds)
                loss = F.nll_loss(F.log_softmax(output, dim=1), label_id)
                loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, epochs:int, comment_to_file_name: str, batch_size, validation_input_ids, validation_labels, evaluator):
    model.train()
    writer = SummaryWriter(comment=comment_to_file_name)
    epoch_loss = 0
    num_training_steps = len(data_loader) * epochs * batch_size
    len_dataloader = len(data_loader)
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(epochs):
        for step, batch in enumerate(data_loader):
            current_step = (step + epoch * len_dataloader) * batch_size
            input = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            labels[labels==0] = -100 
            labels = labels.to(device)
            # print(labels)
            # labels = torch.squeeze(labels, dim=0)
            optimizer.zero_grad()
            output = model(input_ids=input, labels=labels, attention_mask=attention_mask)
            loss = output.loss
            # loss = F.cross_entropy(output, target)
            # epoch_loss += loss.data[0]
            loss.backward()
            # print(loss)
            # batch_size = len(loss)
            # for i, l in enumerate(loss):
            writer.add_scalar("Loss/train", loss.data, current_step)
            optimizer.step()
            progress_bar.update(batch_size)
            if current_step%10000==0:
                eval_result = evaluate(model, validation_input_ids, validation_labels)
                writer.add_scalar("Validation accuracy", eval_result, current_step)
                print(f"Performance on validation set is {eval_result}")
        model_name = "/scratches/dialfs/alta/hln35/model/" + comment_to_file_name
        model.save_pretrained(f"{model_name}_epoch{epoch}")
    writer.flush()
    return model


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float, epochs:int, batch_size, validation_input_ids, validation_labels, comment_to_file_name: str):
    writer = SummaryWriter(comment=comment_to_file_name)
    model.train()
    epoch_loss = 0
    num_training_steps = len(data_loader) * epochs * batch_size
    len_dataloader = len(data_loader)
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(epochs):
        for step, batch in enumerate(data_loader):
            current_step = (step + epoch * len_dataloader) * batch_size
            input = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            output = model(input_ids=input, labels=labels)
            #print(output.loss)
           
            
            loss = output.loss + importance * ewc.penalty(model)
            writer.add_scalar("Loss/train", loss.data, current_step)
            
            #print(loss)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()
            progress_bar.update(batch_size)
            if current_step%50000==0:
                eval_result = evaluate(model, validation_input_ids, validation_labels)
                writer.add_scalar("Validation accuracy", eval_result, current_step)
                print(f"Performance on validation set is {eval_result}")
            
        model_name = "/scratches/dialfs/alta/hln35/model/" + comment_to_file_name
        model.save_pretrained(f"{model_name}_epoch{epoch}")

    writer.flush()
    return model
    
# def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
#               ewc: EWC, importance: float, epochs:int, validation_input_ids, validation_labels, model_name):
#     model.train()
#     loss_per_100 = 0
#     acc_per_100 = 0
#     loss_array = []
#     acc_array = []	
#     eval_results = []
#     c = 0
#     num_training_steps = epochs * len(data_loader)
    
#     progress_bar = tqdm(range(num_training_steps))
#     for epoch in range(epochs):
#         for batch in data_loader:
#              labels = batch["labels"].clone().detach().requires_grad_(True).to(device)
#              labels = labels.squeeze()
#              labels = torch.nn.functional.softmax(labels, dim=-1)
#              input = batch["input_ids"].to(device)
#              optimizer.zero_grad()
#              outputs = model(input, decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(device))
#              loss_fct = torch.nn.CrossEntropyLoss()
#              logits = outputs.get("logits")
#              preds = logits[0, 0, tuple(list(ans_id_dict.keys()))]
            
#              if torch.argmax(labels) == torch.argmax(preds):
#                  acc_per_100 += 1
            
#              loss = loss_fct(preds, labels) + importance * ewc.penalty(model)
#              loss_per_100 += loss.item()
#              if c%100==99:
#                  print(loss_per_100)
#                  loss_array.append(loss_per_100)
#                  acc_array.append(acc_per_100)
               
#                  loss_per_100 = 0
#                  acc_per_100 = 0
               
#              loss.backward()
#              optimizer.step()
#              progress_bar.update(1)
#              c += 1
#              if c%10000==9999:
#                  eval_result = evaluate(model, validation_input_ids, validation_labels)
#                  eval_results.append(eval_result)
#                  print(f"Performance on validation set is {eval_result}")
        
#         model.save_pretrained(f"{model_name}_{'{:.0e}'.format(importance)}_epoch{epoch}")
#    return model, loss_array, acc_array, eval_results

def evaluate(model, input_ids, labels, ans_id_dict=ans_id_dict):
    model.eval()
    model_outputs = []
    probability_output = []
    progress_bar = tqdm(input_ids)
    for i in range(0, len(input_ids)):
            # print(input_ids[i])
        
            test_tensor = torch.unsqueeze(torch.tensor(input_ids[i]), 0).to(device)
            preds = model(input_ids=test_tensor, decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id,]]).to(device))      
            preds_prob = []
            for t in ans_id_dict.keys():
                preds_prob.append(torch.nn.functional.softmax(preds.logits, dim=-1)[...,t][0][0].item())
                
            model_outputs.append(index_to_ans[np.argmax(preds_prob)])
            probability_output.append(preds_prob)
            progress_bar.update(1)
    
    result = 0
    for i in range(min(len(model_outputs), len(labels))):
        # print(labels[i], model_outputs[i])
        if model_outputs[i] == labels[i] or ans_to_index[model_outputs[i]] == labels[i]:
            result += 1
    return result

def evaluate_summary(model, tokenizer, data_loader, batch_size, src_field, ref_field, evaluator):
    max_target_length = 128
    if isinstance(evaluator, SumEvaluator):
        print("Using UniEval")
    else:
        if evaluator.name == "rouge":
            print("Using Rouge")
        else:
            print("Not using Rouge")
    model_name = model.config.name_or_path
    random_sentence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. In fermentum posuere urna nec tincidunt praesent. Arcu risus quis varius quam quisque."
    print(model_name)
    model_small = model
    src_list = []
    ref_list = []
    output_list = []
    progress_bar = tqdm(range(len(data_loader)))
    for data in data_loader:
        # print(data)
        ref_list += data[ref_field]
        src_list += data[src_field]
        input_ids = data["input_ids"].to(device)
        
        preds_tokenized = model_small.generate(input_ids, max_new_tokens=max_target_length, do_sample=False) 
        # cprint(f'model: {model_name},time: {datetime.datetime.now()}', color='green')
        input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        #print(input_text)
        preds = tokenizer.batch_decode(preds_tokenized, skip_special_tokens=True)
        preds = [pred if pred != "" else random_sentence  for pred in preds]
        #print(preds)
        output_list += preds
        progress_bar.update(1)

    #src_list = data_loader[src_field]
    #ref_list = data_loader[ref_field]

    if isinstance(evaluator, SumEvaluator):
        num_ans = len(output_list)
        results_small_dict = {}
        all_data = convert_to_json(output_list=output_list, 
                                   src_list=src_list, ref_list=ref_list)
        eval_scores = evaluator.evaluate(all_data)
        for eval_score in eval_scores:
            for key, value in eval_score.items():
                if key not in results_small_dict:
                    results_small_dict[key] = value
                else:
                    results_small_dict[key] += value
        for k, v in results_small_dict.items():
            results_small_dict[k] = v/num_ans
        eval_scores = results_small_dict
    else:
        if evaluator.name == "rouge":
            eval_scores = evaluator.compute(predictions=output_list, references=ref_list)
    
    # model_output = model_evaluator.predict(text_list, batch_size=batch_size)
    print(f"For model {model_name} the average score on the test set is ")
    # print(len(model_output["system_score"]))
    print(eval_scores)
    model_name = model_name.split("/")[-1]
    with open(f"log_eval/log_evaluation_{model_name}.json", "a") as outfile: 
        json.dump(eval_scores, outfile)
    return eval_scores

def evaluate_mt(model, tokenizer, data_loader, batch_size, source_lang, target_lang, model_evaluator):
    model_name = model.config.name_or_path
    max_target_length = 128
    print(model_name)
    model_small = model

    scores_ewc = []
    text_list = []
    progress_bar = tqdm(range(len(data_loader)))
    for data in data_loader:
        ref = data[target_lang]
        src = data[source_lang]
        input_ids = data["input_ids"].to(device)
        
        preds_tokenized = model_small.generate(input_ids, max_new_tokens=max_target_length, do_sample=False) 
        preds = tokenizer.batch_decode(preds_tokenized, skip_special_tokens=True)
#        print(preds)
        for i in range(len(ref)):
           text_list += [{
                "src": src[i],
                "mt": preds[i],
                "ref": ref[i]
            }]
        progress_bar.update(1)
        
    model_output = model_evaluator.predict(text_list, batch_size=batch_size)
    print(f"For model {model_name} the average score on the test set is ")
    # print(len(model_output["system_score"]))
    print(model_output["system_score"])
    eval_scores = {"system_score" : model_output["system_score"]}
    model_name = model_name.split("/")[-1]
    with open(f"log_eval/log_evaluation_{model_name}.json", "a") as outfile: 
        json.dump(eval_scores, outfile)
        
    return model_output

def compute_loss_generate(input_ids, max_new_tokens, model, tokenizer, device, labels=None):
    # test_tensor = torch.tensor([tokenized_summary["train"][0]["input_ids"]]).to(device)
    # input_ids = test_tensor
    decoder_input_ids = tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    # decoder_input_ids = tokenizer("0", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    # print(decoder_input_ids)
    assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"
    
    # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
    outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    
    # get encoded sequence
    encoded_sequence = (outputs.encoder_last_hidden_state,)
    # get logits
    lm_logits = outputs.logits
    # print(lm_logits)
    # sample last token with highest prob
    
    if labels != None:
        next_decoder_input_ids = labels[0][0]
        l = lm_logits[0][-1][next_decoder_input_ids]
        next_decoder_input_ids = torch.unsqueeze(torch.unsqueeze(next_decoder_input_ids,0),0)
    else:
        next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
        l = torch.max(lm_logits[:, -1:])
    # print(next_decoder_input_ids)
    # print(decoder_input_ids)
    decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
    decoder_input_ids = decoder_input_ids.detach()
    # print(l)
    # loss = lm_logits[next_decoder_input_ids, -1:]

    next_decoder_input_ids = "0"
    no_tokens = 1
    if labels != None:
        while next_decoder_input_ids and no_tokens<len(labels[0]):
            lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
            next_decoder_input_ids = labels[0][0+no_tokens]            
            l = torch.add(l,lm_logits[0][-1][next_decoder_input_ids])
            next_decoder_input_ids = torch.unsqueeze(torch.unsqueeze(next_decoder_input_ids,0),0)
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            no_tokens += 1
    else:
        while next_decoder_input_ids and next_decoder_input_ids != 1 and no_tokens<=max_new_tokens:
            lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
            l = torch.add(l,torch.max(lm_logits[:, -1:]))
            # print(l)
            # sample last token with highest prob again
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            # loss += lm_logits[next_decoder_input_ids, -1:]
            # concat again
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            no_tokens += 1
            # print(lm_logits)
    # print(decoder_input_ids)
    l.backward()
    print(l)
    del l

def flatten_params(matrix):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(matrix[p]) for p in matrix]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    # index_list = torch.arange(flat.shape[0], device=device).view(-1,1)
    # flat = torch.cat((flat, index_list), -1) 
    
    return {"params": flat, "indices": indices}

def recover_flattened(flat_params, indices, model_dict):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    # print(l)
    # print(model_dict)
    for i, p in enumerate(model_dict.values()):
        print(p.shape)
    for i, p in enumerate(model_dict.values()):
        print(i)
        l[i] = l[i].view(*p.shape)
    return l
    
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def em_evaluator(predictions, ground_truths, xlingual=False):
    res = 0
    for i in range(len(predictions)):
        if exact_match(predictions[i], ground_truths[i], xlingual):
            res += 1
    return res

def pad_dataset(data_points, tokenizer, max_tokens_output_len):
    pad_input_ids = tokenizer("<pad>", add_special_tokens=False).input_ids[0]
    
    if len(data_points["label_ids"]) < max_tokens_output_len:
        data_points["label_ids"] += [pad_input_ids] * (max_tokens_output_len - len(data_points["label_ids"]))
    elif len(data_points["label_ids"]) > max_tokens_output_len:
        data_points["label_ids"] = data_points["label_ids"][:max_tokens_output_len]
        
    return data_points

def process_data_nat_inst(task_name, task_length=100):
    with open(task_name, "r") as read_content: 
        fields = json.load(read_content)
        fields.pop("Instances")
    dataset = load_dataset('json', data_files=task_name, field="Instances")
    # task_length = len(dataset["train"])
    dataset_formatted = Dataset.from_dict(dataset["train"][:task_length])
    fields["Instances"] = dataset_formatted
    return fields

def preprocess_function_translate(examples, source_lang, target_lang, max_input_length, max_target_length):
    prefix_translate = "translate English to French: "
    inputs = [prefix_translate + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    # print(inputs[0], targets[0])
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True, )
    # model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_summary(examples, max_input_length, max_target_length):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(examples["summary"], max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_race(data_points, max_input_length):
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
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    labels = tokenizer(data_points["answer"], max_length=2, padding="max_length", truncation=True)
    # model_inputs["label"] = list(map(lambda x: int(ans_to_index[x]), data_points["answer"]))
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs