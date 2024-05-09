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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import itertools


model_small = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_small)

class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: tokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['text']

        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = tokenizer(
            data_row['summary'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation

        return dict(
            text=text,
            summary=data_row['summary'],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )
    
class GeneralDataModule(pl.LightningDataModule):
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

    # glue_task_num_labels = {
    #     "cola": 2,
    #     "sst2": 2,
    #     "mrpc": 2,
    #     "qqp": 2,
    #     "stsb": 1,
    #     "mnli": 3,
    #     "qnli": 2,
    #     "rte": 2,
    #     "wnli": 2,
    #     "ax": 3,
    # }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    
    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        input_max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
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
        self.dataset = self.task_dataset_split_map[task_name]
        self.text_fields = self.task_text_field_map[task_name]
        self.prompts = self.task_prompt_map[task_name]
        self.label_field = self.task_label_field[task_name][0]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage=None):

        self.dataset = datasets.load_dataset(self.dataset)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" or "test" in x]
        

        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def prepare_data(self):
        datasets.load_dataset(self.dataset)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        for i in range(len(self.prompts)):
            texts = zip(itertools.repeat(self.prompts[i]))
            if i < len(self.text_fields) - 1:
                texts = zip(texts, example_batch[self.text_fields[i]])
        
        texts = list(texts)

        # if len(self.text_fields) > 1:
        #     texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # else:
        #     texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.input_max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch[self.label_field]

        return features
N_EPOCHS = 3
BATCH_SIZE = 8

data_module = NewsSummaryDataModule(train_df, test_df, tokenizer)

class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_small, return_dict=True)
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
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
    
    def validation_step(self, batch, batch_size):
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

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
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
        return AdamW(self.parameters(), lr=0.0001)
    

model = NewsSummaryModel()
     
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

logger = TensorBoardLogger("lightning_logs", name='news-summary')

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)