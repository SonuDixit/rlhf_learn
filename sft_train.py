# imports
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
# get dataset
dataset = load_dataset("imdb", split="train")

training_args = TrainingArguments(num_train_epochs=1,
                                  output_dir='./')
# get trainer

trainer = SFTTrainer(
    "facebook/opt-350m",
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=32,
    )

# train
trainer.train()