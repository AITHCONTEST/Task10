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


def load_tokenizer_and_model(path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        add_bos_token=False,
    )
    gemma_path = "/" + os.path.join(*path, "gemma")
    gemma_adapter = adapter_path
    gemma_pretrained = "/" + os.path.join(*path, "gemma_pretrained")
    if os.path.exists(gemma_pretrained):
        model = AutoModelForCausalLM.from_pretrained(
            gemma_pretrained,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        return tokenizer, model

    os.makedirs(gemma_pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, gemma_adapter)
    model = model.merge_and_unload()
    model.save_pretrained(
        save_directory=gemma_pretrained,
    )
    return tokenizer, model


@hydra.main(config_path="config", config_name="config")
def launch(cfg):
    final_model_dir = os.path.join(HydraConfig.get().runtime.output_dir, "finalModel")
    transformers.set_seed(cfg.seed)
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    with open(
        "/" + os.path.join(*final_model_dir.split("/")[:-4], "train/dataTranslate.json")
    ) as json_data:
        data = json.load(json_data)

    train = Dataset.from_dict({"content": data["train"]})
    test = Dataset.from_dict({"content": data["test"]})
    raw_datasets = DatasetDict()
    raw_datasets["train"] = train
    raw_datasets["test"] = test
    context_length = 2048
    tokenizer, model = load_tokenizer_and_model(
        final_model_dir.split("/")[:-4],
        # '/home/jovyan/pipe_t5_mt/outputs/2024-09-12/01-50-39/finalModel'
        "/home/maxim/Documents/projects/Task10/outputs/2024-09-12/11-13-54/pretrain/checkpoint-1",
    )
    tokenizer.padding_side = "right"

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Gemma size: {model_size/1000**2:.1f}M parameters")

    TRANSLATE = "<TRANSLATE>"
    COMPUTE = "<COMPUTE>"
    response_template = TRANSLATE + 50 * COMPUTE
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
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
    args = SFTConfig(
        max_seq_length=context_length,
        dataset_text_field="content",
        output_dir="pretrain",
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

    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    trainer.save_model(output_dir=final_model_dir)


if __name__ == "__main__":
    launch()
