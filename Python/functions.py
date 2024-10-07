import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import transformers
import sklearn
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Data Augmentation imports
import nltk
from nltk.corpus import wordnet

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    log_level: str = field(default="info")
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    
    def get_process_log_level(self):
        return 10
    
    def get_warmup_steps(self, num_training_steps):
        return 8

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    use_augmentation: bool = field(default=False, metadata={"help": "Whether to use data augmentation"})

class SequenceDataAugmenter:
    def __init__(self, mask_prob=0.15, shuffle_prob=0.1, synonym_replace_prob=0.1):
        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob
        self.synonym_replace_prob = synonym_replace_prob
        nltk.download('wordnet', quiet=True)

    def random_masking(self, sequence):
        words = sequence.split()
        masked_words = [word if random.random() > self.mask_prob else '[MASK]' for word in words]
        return ' '.join(masked_words)

    def sequence_shuffling(self, sequence):
        words = sequence.split()
        if random.random() < self.shuffle_prob:
            random.shuffle(words)
        return ' '.join(words)

    def synonym_replacement(self, sequence):
        words = sequence.split()
        new_words = []
        for word in words:
            if random.random() < self.synonym_replace_prob:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                if len(synonyms) > 0:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return ' '.join(new_words)

    def augment(self, sequence):
        augmented = self.random_masking(sequence)
        augmented = self.sequence_shuffling(augmented)
        augmented = self.synonym_replacement(augmented)
        return augmented

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1,
                 use_augmentation: bool = False):

        super(SupervisedDataset, self).__init__()

        self.augmenter = SequenceDataAugmenter() if use_augmentation else None

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
            random.shuffle(data)
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        
        if self.augmenter:
            augmented_texts = []
            for text in texts:
                if isinstance(text, list):
                    augmented_texts.append([self.augmenter.augment(t) for t in text])
                else:
                    augmented_texts.append(self.augmenter.augment(text))
            texts.extend(augmented_texts)
            labels.extend(labels)  # Duplicate labels for augmented data
        
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def predictionAccuracyMetrics(yobs, ypred, mode):
    if mode == "classification":
        ypred_bin = np.copy(ypred)
        ypred_bin[ypred_bin < 0.5] = 0
        ypred_bin[ypred_bin >= 0.5] = 1
        AUROC = np.round(roc_auc_score(yobs, ypred), 4)
        AUPR = np.round(average_precision_score(yobs, ypred), 4)
        F1 = np.round(f1_score(yobs, ypred_bin), 4)
        MCC = np.round(matthews_corrcoef(yobs, ypred_bin > 0.5), 4)
        metrics = pd.DataFrame({'AUROC': [AUROC], 'AUPR': [AUPR], 'MCC': [MCC]})       

    elif mode == "regression":
        yobs = yobs[:, 0]
        ypred = ypred[:, 0]
        idxIso = range(0, int(len(yobs) * 0.1))
        idxVal = range(int(len(yobs) * 0.1) + 1, len(yobs))
        isoRegCalibration = IsotonicRegression()
        isoRegCalibration.fit(ypred[idxIso], yobs[idxIso])
        yprediso = isoRegCalibration.predict(ypred[idxVal])
        yprediso[np.isnan(yprediso)] = 0
        R = np.round(np.corrcoef(yobs[idxVal], ypred[idxVal])[0, 1], 4)
        Riso = np.round(np.corrcoef(yobs[idxVal], yprediso)[0, 1], 4)     
        metrics = pd.DataFrame({'R': [R], 'R_isotonic': [Riso]})        
            
    return metrics

# This function is mentioned but not implemented in the original code
# You may want to implement it or remove the reference if not needed
def load_or_generate_kmer(data_path, texts, kmer):
    # Implementation goes here
    pass