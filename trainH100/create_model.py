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


def prepare_tokenizer(NEW_TOKENS_PATH):
    # tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    with open(NEW_TOKENS_PATH, "r+") as file1:
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
        "additional_special_tokens": [
            RUS,
            UDM,
            KOMI,
            MHR,
            MANS,
            TRANSLATE,
            COMPUTE,
            VOCAB,
        ]
    }
    # vocab.extend([RUS, UDM, KOMI, MHR, MANS])
    new_tokens = set(vocab) - set(tokenizer.vocab.keys())
    new_tokens = sorted(list(new_tokens))
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


@hydra.main(config_path="config", config_name="config")
def create_model_and_tokenizer(cfg):
    tokenizer = prepare_tokenizer(cfg.NEW_TOKENS_PATH)
    if os.path.exists(cfg.GEMMA_PATH):
        return

    os.makedirs(cfg.GEMMA_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        # device_map="auto",
        # torch_dtype=torch.bfloat16,
        # attn_implementation="eager",
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     "google/gemma-2-9b",
    #     torch_dtype=torch.bfloat16,
    # )
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(
        save_directory=cfg.GEMMA_PATH,
    )
    tokenizer.save_pretrained(
        save_directory=cfg.GEMMA_PATH,
    )

    peft_config = LoraConfig(
        r=512,
        lora_alpha=512,
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
    if os.path.exists(cfg.GEMMA_PEFT_PATH):
        return

    os.makedirs(cfg.GEMMA_PEFT_PATH)
    model.save_pretrained(
        save_directory=cfg.GEMMA_PEFT_PATH,
    )


if __name__ == "__main__":
    create_model_and_tokenizer()
