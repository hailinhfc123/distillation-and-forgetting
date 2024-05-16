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
from comet import download_model, load_from_checkpoint
from ewc_utils import preprocess_function_translate, evaluate_mt



cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'
model_small = "google/flan-t5-small"
if torch.cuda.is_available() == False:
    raise Exception("Cuda is not available, please enable cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_small)
model = AutoModelForSeq2SeqLM.from_pretrained(model_small).to(device)


# model_path = download_model("Unbabel/wmt22-comet-da", saving_directory="/scratches/dialfs/alta/hln35/.cache", local_files_only=True)
model_path = download_model("Unbabel/wmt22-comet-da", saving_directory="/scratches/dialfs/alta/hln35/.cache")

model_evaluator = load_from_checkpoint(model_path)

# raw_datasets = load_dataset("xsum", cache_dir=cache_dir)
# task = 'summarization'

max_input_length = 512
max_target_length = 128
source_lang = "en"
target_lang = "fr"

    
model_list = ["/scratches/dialfs/alta/hln35/model/flant5_small_distill_xsum_batchsize_1_10k_samples_epoch0", 
              "/scratches/dialfs/alta/hln35/model/flant5_small_distill_xsum_batchsize_1_10k_samples_epoch1", 
              "/scratches/dialfs/alta/hln35/model/flant5_small_distill_xsum_batchsize_1_10k_samples_epoch2",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_10k_samples_epoch0",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_10k_samples_epoch1",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_10k_samples_epoch2",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_full_samples_epoch0",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_full_samples_epoch1",
              "/scratches/dialfs/alta/hln35/model/flant5_small_finetune_xsum_batchsize_4_full_samples_epoch2",
             ]
batch_size = 8
translate_datapoints = load_dataset("presencesw/wmt14_fr_en", split="test", cache_dir=cache_dir)
tokenized_translate = translate_datapoints.map(lambda b: preprocess_function_translate(b, source_lang, target_lang, max_input_length, max_target_length), 
                                               batched=True)
tokenized_translate.set_format("torch")
test_translate_set = tokenized_translate
# test_translate_set = tokenized_translate.set_format("torch")
test_translate_dataloader = DataLoader(test_translate_set, batch_size=batch_size)

for model in model_list:
    model_small_trained = model
    model_small_trained = AutoModelForSeq2SeqLM.from_pretrained(model_small_trained, local_files_only=True).to(device)
    test_scores = evaluate_mt(model=model_small_trained, tokenizer=tokenizer, data_loader=test_translate_dataloader,
                batch_size=batch_size, source_lang=source_lang, target_lang=target_lang)
    # print("score is ", test_scores)
    #model_small_ewc = AutoModelForSeq2SeqLM.from_pretrained(model, local_files_only=True).to(device)

# model_list = ["google/flan-t5-small", "google/flan-t5-large", "/scratches/dialfs/alta/hln35/distillation/model/flant5_small_lr_10-4_race_finetuning_epoch2", "/scratches/dialfs/alta/hln35/distillation/model/flant5_small_lr_10-4_race_distill_epoch2"]
# for importance in [1e-0, 1e-4, 1e-2]:
#     for epoch in [2, 1, 0]:
 
# for model in model_list:
#         model_name = model
#         # model_small_ewc = f"/scratches/dialfs/alta/hln35/model/flant5_small_lr_10-4_race_ewc_after_translation_importance_{'{:.0e}'.format(importance)}_epoch{epoch}"
#         model_small_ewc = model
#         print(model_small_ewc)
#         model_small_ewc = AutoModelForSeq2SeqLM.from_pretrained(model_small_ewc, local_files_only=True).to(device)
#         #model_small_ewc = AutoModelForSeq2SeqLM.from_pretrained(model, local_files_only=True).to(device)
#         scores_ewc = []
#         progress_bar = tqdm(range(len(books)))
#         for i in range(len(books)):
#             text = prefix + books[i]["translation"][source_lang]
#             ref = books[i]["translation"][target_lang]
#             inputs = tokenizer(text, return_tensors="pt").to(device).input_ids
#             preds_tokenized = model_small_ewc.generate(inputs, max_new_tokens=128, do_sample=False) 
#             preds = tokenizer.batch_decode(preds_tokenized, skip_special_tokens=True)
#             data = [
#                 {
#                     "src": books[i]["translation"][source_lang],
#                     "mt": preds[0],
#                     "ref": ref
#                 }
#             ]
#             model_output = model_evaluator.predict(data, batch_size=1)
#             # print (model_output)
#             # print(data)
#             scores_ewc.append(model_output["system_score"])
#             progress_bar.update(1)
#         #print(f"For model {model} the average score on the test set is {sum(scores_ewc)/len(books)}")
#         print(f"For importance {importance}, epoch {epoch} the average score on the test set is {sum(scores_ewc)/len(books)}")
