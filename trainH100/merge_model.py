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


@hydra.main(config_path="config", config_name="config")
def merge_model(cfg):
    if os.path.exists(cfg.GEMMA_PRETRAINED_PATH):
        return

    os.makedirs(cfg.GEMMA_PRETRAINED_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.GEMMA_PATH,
        device_map="auto",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, cfg.FINAL_MODEL_PRETRAINED)
    model = model.merge_and_unload()
    model.save_pretrained(
        save_directory=cfg.GEMMA_PRETRAINED_PATH,
    )
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
    if os.path.exists(cfg.GEMMA_PRETRAINED_LORA_PATH):
        return

    os.makedirs(cfg.GEMMA_PRETRAINED_LORA_PATH)
    model.save_pretrained(
        save_directory=cfg.GEMMA_PRETRAINED_LORA_PATH,
    )


if __name__ == "__main__":
    merge_model()
