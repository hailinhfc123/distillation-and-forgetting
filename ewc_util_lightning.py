import lightning as L
from copy import deepcopy
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.optim import AdamW
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
import json, re
import string
# import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
import itertools
from collections import defaultdict
from typing import Optional
import evaluate
from datetime import datetime

cache_dir = "/scratches/dialfs/alta/hln35/.cache"
os.environ['TRANSFORMERS_CACHE'] = '/scratches/dialfs/alta/hln35/.cache'
# cache_dir = "./.cache"
# os.environ['TRANSFORMERS_CACHE'] = './.cache'
model_small = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_small)

# class NewsSummaryDataset(Dataset):
#     def __init__(
#         self,
#         data: pd.DataFrame,
#         tokenizer: tokenizer,
#         text_max_token_len: int = 512,
#         summary_max_token_len: int = 128
#     ):
#         self.tokenizer = tokenizer
#         self.data = data
#         self.text_max_token_len = text_max_token_len
#         self.summary_max_token_len = summary_max_token_len
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index: int):
#         data_row = self.data.iloc[index]

#         text = data_row['text']

#         text_encoding = tokenizer(
#             text,
#             max_length=self.text_max_token_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors='pt'
#         )

#         summary_encoding = tokenizer(
#             data_row['summary'],
#             max_length=self.summary_max_token_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors='pt'
#         )

#         labels = summary_encoding['input_ids']
#         labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation

#         return dict(
#             text=text,
#             summary=data_row['summary'],
#             text_input_ids=text_encoding['input_ids'].flatten(),
#             text_attention_mask=text_encoding['attention_mask'].flatten(),
#             labels=labels.flatten(),
#             labels_attention_mask=summary_encoding['attention_mask'].flatten()
#         )
    
class GeneralDataModule(L.LightningDataModule):
    task_prompt_map = {
        "xsum": ["summarise: "],
        "wmt14": ["translate English to French: "],
        # "race": ["answer this question by choosing the best choice either A, B, C, or D. Given the context is: ", ". Choices: "],
        "boolq": ['Paragraph: ', "answer this question by choosing the best choice either A, B. Based on the paragraph above: ", ". Options: A. False \n B. True"],
        "sst2": ['Paragraph: ', "answer this question by choosing the best choice either A, B. Based on the paragraph above can we conlude that is positive or negative. Options: A. Negative \n B. Positive"],
        "anli": ['Paragraph: ', ". Answer this question by choosing the best choice either A, B, C. Based on the paragraph above can we conclude that ", ". Options: A. Yes \n B. Not possible \n C. No"],
        "mnli": ['Paragraph: ', ". Answer this question by choosing the best choice either A, B, C. Based on the paragraph above can we conclude that ", ". Options: A. Entailment \n B. Neutral \n C. Contradiction"]
    }
    task_text_field_map = {
        "xsum": ["document"],
        "wmt14": ["en"],
        # "race": ["answer this question by choosing the best choice either A, B, C, or D. Given the context is: ", ". Choices: "],
        "boolq": ["passage", "question"],
        "anli": ["premise", "hypothesis"],
        "sst2": ["sentence"],
        "mnli": ["text1", "text2"],
    }

    task_label_field = {
        "xsum": ["summary"],
        "wmt14": ["fr"],
        # "race": ["answer this question by choosing the best choice either A, B, C, or D. Given the context is: ", ". Choices: "],
        "boolq": ["answer"],
        "anli": ["label"],
        "sst2": ["label"],
        "mnli": ["label"],
    }

    task_dataset_split_map = {
        "xsum": ["EdinburghNLP/xsum"],
        "wmt14": ["presencesw/wmt14_fr_en"],
        # "race": ["answer this question by choosing the best choice either A, B, C, or D. Given the context is: ", ". Choices: "],
        "boolq": ["google/boolq"],
        "anli": ["facebook/anli"],
        "sst2": ["stanfordnlp/sst2"],
        "mnli": ["SetFit/mnli"],   
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
        "labels_attention_mask",
        "sources",
    ]
    
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "xsum",
        input_max_seq_length: int = 512,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        output_max_seq_length: int = 128,
        **kwargs
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.input_max_seq_length = input_max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.output_max_seq_length = output_max_seq_length
        self.dataset = self.task_dataset_split_map[task_name][0]
        self.text_fields = self.task_text_field_map[task_name]
        self.prompts = self.task_prompt_map[task_name]
        self.label_field = self.task_label_field[task_name][0]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage=None):

        self.dataset = datasets.load_dataset(self.dataset, cache_dir=cache_dir)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        self.test_splits = [x for x in self.dataset.keys() if "test" in x]

    def prepare_data(self):
        datasets.load_dataset(self.dataset, cache_dir=cache_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]
        else:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)


    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.test_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.test_splits]
        else:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)


    def text_and_prompt_generator(self, zipped_text_and_promt):
        for t in zipped_text_and_promt:
            yield {"complete_text" : "".join(t)}

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        len_dataset = len(example_batch[self.text_fields[0]])
        iterator_list = []
        texts = None
        for i in range(len(self.prompts)):
            iterator_list.append(itertools.repeat(self.prompts[i], len_dataset))
            # texts = itertools.repeat(self.prompts[i], len_dataset) if texts == None else zip(texts, itertools.repeat(self.prompts[i], len_dataset))
            if i < len(self.text_fields) - 1:
                iterator_list.append(example_batch[self.text_fields[i]])
                # texts = zip(texts, example_batch[self.text_fields[i]])
        texts = zip(*iterator_list)

        dataset = Dataset.from_generator(lambda : self.text_and_prompt_generator(texts))
        # texts = list(texts)

        # if len(self.text_fields) > 1:
        #     texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # else:
        #     texts_or_text_pairs = example_batch[self.text_fields[0]]
        complete_texts = dataset["complete_text"]
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            dataset["complete_text"], max_length=self.input_max_seq_length, pad_to_max_length=True, truncation=True
        )
        labels = self.tokenizer.batch_encode_plus(
            example_batch[self.label_field], max_length=self.output_max_seq_length, pad_to_max_length=True, truncation=True)
        print(features)
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = labels["input_ids"]
        features["labels_attention_mask"] = labels["attention_mask"]
        features["sources"] = example_batch[self.text_fields[0]]

        return features
    


# data_module_xsum = GeneralDataModule(model_small)
# data_module_anli = GeneralDataModule(model_small, task_name="anli")


class NewModel(L.LightningModule):
    task_evaluator_map = {
        "xsum": "rouge",
        "wmt14": "bleu",
        # "race": ["answer this question by choosing the best choice either A, B, C, or D. Given the context is: ", ". Choices: "],
        "boolq": "accuracy",
        "anli": "accuracy",
        "sst2": "accuracy",
        "mnli": "accuracy",
    }
    def __init__(self,
                model_name_or_path: str,
                task_name: str,
                learning_rate: float = 1e-4,
                adam_epsilon: float = 1e-6,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                train_batch_size: int = 8,
                eval_batch_size: int = 8,
                eval_splits: Optional[list] = None,
                **kwargs,):
        super().__init__()
        self.save_hyperparameters()

        self.metric = evaluate.load(
            self.task_evaluator_map[self.hparams.task_name], experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.outputs = defaultdict(list)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        sources = batch['']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        if self.task_evaluator_map[self.hparams.task_name] == "accuracy":
            preds = outputs[:, 1]
        else:
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.outputs[dataloader_idx].append({"loss": loss, "preds": preds, "labels": labels})
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
    
    def on_validation_epoch_end(self):
        # if self.hparams.task_name == "mnli":
        #     for i, outputs in self.outputs.items():
        #         # matched or mismatched
        #         split = self.hparams.eval_splits[i].split("_")[-1]
        #         preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        #         labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        #         loss = torch.stack([x["loss"] for x in outputs]).mean()
        #         self.log(f"val_loss_{split}", loss, prog_bar=True)
        #         split_metrics = {
        #             f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
        #         }
        #         self.log_dict(split_metrics, prog_bar=True)
        #     return loss

        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)
        if self.task_evaluator_map[self.hparams.task_name] == "accuracy":
            preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
            loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        else:
            preds = torch.cat([x["preds"] for x in flat_outputs])
            labels = torch.cat([x["labels"] for x in flat_outputs])
            loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
            self.log("val_loss", loss, prog_bar=True)
            # if self.task_evaluator_map[self.hparams.task_name] == "comet":
            #     self.log_dict(self.metric.compute(predictions=preds, references=labels, sources=), prog_bar=True)
            self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.outputs.clear()
    
N_EPOCHS = 3
BATCH_SIZE = 1

checkpoint_callback = ModelCheckpoint(
    dirpath = cache_dir + '/new_model_checkpoints',
    filename='best-checkpoint-{epoch}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    auto_insert_metric_name=True
)

logger = TensorBoardLogger(save_dir=cache_dir + '/log', version=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), name="lightning_logs")

seed_everything(42, workers=True)

trainer = L.Trainer(
    accelerator="cpu",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=N_EPOCHS,
    deterministic=True,
)

dm_mnli = GeneralDataModule(
    model_name_or_path=model_small,
    task_name="xsum",
    eval_batch_size=BATCH_SIZE,
    train_batch_size=BATCH_SIZE,
)
dm_mnli.setup("fit")

model = NewModel(
    model_name_or_path=model_small,
    eval_splits=dm_mnli.eval_splits,
    task_name=dm_mnli.task_name,
    eval_batch_size=BATCH_SIZE,
    train_batch_size=BATCH_SIZE,
)

trainer.validate(model, dm_mnli)

trainer.fit(model, dm_mnli)
