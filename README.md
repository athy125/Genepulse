# Genepulse 
 Mixtral-8x7B streamlined for DNA analysis

<<<<<<< HEAD

## Project Overview

This project implements a  machine learning pipeline for sequence classification tasks. It leverages state-of-the-art transformer models and includes advanced features such as data augmentation, distributed training support, and comprehensive evaluation metrics.
=======
## Project Overview

This project implements a machine learning pipeline for sequence classification tasks. It leverages state-of-the-art transformer models and includes advanced features such as data augmentation, distributed training support, and comprehensive evaluation metrics.
>>>>>>> 703d384abad4832c083f735f0348c0c378faecff

### Key Features

- Flexible support for single sequence and sequence-pair classification tasks
- Custom data augmentation techniques for NLP tasks
- Support for Low-Rank Adaptation (LoRA) fine-tuning
- Comprehensive evaluation metrics for both classification and regression tasks

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.0+
- scikit-learn
- pandas
- nltk

### Basic Training

To train a model on your dataset:

```python
from sequence_classification import SupervisedDataset, DataCollatorForSupervisedDataset, TrainingArguments, ModelArguments, DataArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Initialize arguments
model_args = ModelArguments(model_name_or_path="bert-base-uncased")
data_args = DataArguments(data_path="path/to/your/data.csv", use_augmentation=True)
training_args = TrainingArguments(output_dir="./output", num_train_epochs=3)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)

# Prepare dataset
dataset = SupervisedDataset(data_args.data_path, tokenizer, use_augmentation=data_args.use_augmentation)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
```

### Advanced Usage

#### Enabling LoRA

To use LoRA for efficient fine-tuning:

```python
model_args = ModelArguments(
    model_name_or_path="bert-base-uncased",
    use_lora=True,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="query,value"
)
```

#### Custom Data Augmentation

You can customize the data augmentation process by modifying the `SequenceDataAugmenter` class:

```python
class CustomSequenceDataAugmenter(SequenceDataAugmenter):
    def __init__(self, mask_prob=0.2, shuffle_prob=0.15, synonym_replace_prob=0.1):
        super().__init__(mask_prob, shuffle_prob, synonym_replace_prob)
    
    # Add or override augmentation methods as needed
```

## Project Structure

- `model_arguments.py`: Contains `ModelArguments` for model configuration.
- `training_arguments.py`: Extends Hugging Face's `TrainingArguments` with custom options.
- `data_arguments.py`: Defines arguments for data processing and augmentation.
- `dataset.py`: Implements `SupervisedDataset` for data loading and preprocessing.
- `data_collator.py`: Contains `DataCollatorForSupervisedDataset` for batch preparation.
- `augmentation.py`: Implements `SequenceDataAugmenter` for NLP-specific data augmentation.
- `metrics.py`: Defines functions for calculating various evaluation metrics.
- `train.py`: Main script for training and evaluation.

## Advanced Components

### Data Augmentation

The `SequenceDataAugmenter` class provides three main augmentation techniques:

1. Random Masking: Randomly replaces words with a [MASK] token.
2. Sequence Shuffling: Randomly shuffles the order of words in a sequence.
3. Synonym Replacement: Replaces words with their synonyms using WordNet.

These techniques can be applied individually or in combination to increase dataset diversity and improve model robustness.

<!-- ### Distributed Training

The project includes basic support for distributed training using PyTorch's DistributedDataParallel. To enable distributed training, use the appropriate PyTorch distributed launch command:

```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py
``` -->

### Evaluation Metrics

The project provides comprehensive evaluation metrics for both classification and regression tasks:

- Classification: Accuracy, F1 Score, Matthews Correlation Coefficient, Precision, Recall, AUROC, AUPR
- Regression: Pearson Correlation Coefficient, Isotonic Regression Calibrated Correlation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
