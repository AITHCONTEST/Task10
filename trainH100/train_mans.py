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
from peft import LoraConfig, get_peft_model, PeftModel

# from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def launch(cfg):
    final_model_dir = os.path.join(HydraConfig.get().runtime.output_dir, "finalModel")
    transformers.set_seed(cfg.seed)
    with open(
        "/"
        + os.path.join(*final_model_dir.split("/")[:-4], "trainH100/dataTranslate.json")
    ) as json_data:
        data = json.load(json_data)

    train = Dataset.from_dict({"content": data["train"]})
    test = Dataset.from_dict({"content": data["test"]})
    raw_datasets = DatasetDict()
    raw_datasets["train"] = train
    raw_datasets["test"] = test
    context_length = 2048

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.GEMMA_PATH,
        add_bos_token=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.GEMMA_PRETRAINED_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        model,
        cfg.GEMMA_PRETRAINED_LORA_PATH,
        is_trainable=True,
    )
    tokenizer.padding_side = "right"

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Gemma size: {model_size/1000**2:.1f}M parameters")
    model.print_trainable_parameters()

    TRANSLATE = "<TRANSLATE>"
    COMPUTE = "<COMPUTE>"
    response_template = TRANSLATE + 50 * COMPUTE
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    args = SFTConfig(
        max_seq_length=context_length,
        dataset_text_field="content",
        output_dir="train_mans",
        per_device_train_batch_size=cfg.device_train_batch_size,
        per_device_eval_batch_size=cfg.device_eval_batch_size,
        evaluation_strategy="steps",
        # optim="paged_adamw_32bit",
        eval_steps=50,
        logging_steps=5,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        save_steps=50,
        bf16=True,
        save_safetensors=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=collator,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
    )

    trainer.train()

    trainer.save_model(output_dir=cfg.FINAL_MODEL)


if __name__ == "__main__":
    launch()
