import hydra
from hydra.core.hydra_config import HydraConfig

import os
import json
import copy
import torch
import logging

from torch import nn
from datasets import load_dataset, DatasetDict, Dataset
import transformers
from peft import LoraConfig, get_peft_model

# from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

log = logging.getLogger(__name__)


def create_model(tokenizer, path_model):
    gemma_path = "/" + os.path.join(*path_model, "gemma")
    if os.path.exists(gemma_path):
        model = AutoModelForCausalLM.from_pretrained(
            gemma_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="eager",
        )
        return model
    
    os.makedirs(gemma_path)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="eager",
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     "google/gemma-2-9b",
    #     torch_dtype=torch.bfloat16,
    # )
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(
        save_directory=gemma_path,
    )
    return model


def prepare_tokenizer(path):
    # tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    with open("/" + os.path.join(*path, "train/spm_man.vocab"), "r+") as file1:
        vocab = [elem.split("\t")[0] for elem in file1.read().split("\n")]

    RUS = "<RUS>"
    UDM = "<UDM>"
    KOMI = "<KOMI>"
    MHR = "<MHR>"
    MANS = "<MANS>"
    TRANSLATE = "<TRANSLATE>"
    COMPUTE = "<COMPUTE>"
    VOCAB = "<VOCAB>"
    special_tokens_dict = {
        "additional_special_tokens": [RUS, UDM, KOMI, MHR, MANS, TRANSLATE, COMPUTE, VOCAB]
    }
    # vocab.extend([RUS, UDM, KOMI, MHR, MANS])
    new_tokens = set(vocab) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


@hydra.main(config_path="config", config_name="config")
def launch(cfg):
    final_model_dir = os.path.join(HydraConfig.get().runtime.output_dir, "finalModel")
    transformers.set_seed(cfg.seed)
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    with open(
        "/" + os.path.join(*final_model_dir.split("/")[:-4], "train/data.json")
    ) as json_data:
        data = json.load(json_data)

    train = Dataset.from_dict({"content": data["train"]})
    test = Dataset.from_dict({"content": data["test"]})
    raw_datasets = DatasetDict()
    raw_datasets["train"] = train
    raw_datasets["test"] = test
    context_length = 2048
    tokenizer = prepare_tokenizer(final_model_dir.split("/")[:-4])

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            add_special_tokens=False,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length <= context_length and length >= 6:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    del raw_datasets
    print(len(tokenized_datasets["train"]))
    print(len(tokenized_datasets["test"]))

    model = create_model(tokenizer, final_model_dir.split("/")[:-4])
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Gemma size: {model_size/1000**2:.1f}M parameters")

    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    peft_config = LoraConfig(
        r=256,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
    )
    model = get_peft_model(model, peft_config)
    args = TrainingArguments(
        output_dir="pretrain",
        per_device_train_batch_size=cfg.device_train_batch_size,
        per_device_eval_batch_size=cfg.device_eval_batch_size,
        evaluation_strategy="steps",
        # optim="paged_adamw_32bit",
        eval_steps=2,
        logging_steps=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        save_steps=1,
        bf16=True,
        save_safetensors=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     args=args,
    #     peft_config=peft_config,
    #     data_collator=data_collator,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["test"],
    #     tokenizer=tokenizer,
    #     dataset_text_field="input_ids",
    #     packing=False,
    # )
    trainer.train()

    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    trainer.save_model(output_dir=final_model_dir)


if __name__ == "__main__":
    launch()
