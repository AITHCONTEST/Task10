import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import numpy as np
import pickle 
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers import get_constant_schedule_with_warmup
from transformers.optimization import Adafactor
import random
import re
import sys
import typing as tp
import unicodedata
from sacremoses import MosesPunctNormalizer


mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


# Initialize distributed process group
def setup():
    dist.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    random.seed(rank)
    return rank

# Cleanup function for distributed training
def cleanup():
    dist.destroy_process_group()

LANGS = [('RUS', 'rus_Cyrl'), ('MANS', 'man_Cyrl')]

def get_batch_pairs(batch_size, data):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2


def get_batch_pairs_vocab(batch_size, data):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    sub_data = data[data.src_lang==l1]
    for _ in range(batch_size):
        item = sub_data.iloc[random.randint(0, len(sub_data)-1)]
        xx.append(preproc(item["source"]))
        yy.append(preproc(item["target"]))
    return xx, yy, long1, long2

def fix_tokenizer(tokenizer, new_lang='man_Cyrl'):
    """
    Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer._additional_special_tokens.append("<VOCAB>")
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

# Training function
def train():
    rank = setup()

    tokenizer = NllbTokenizer.from_pretrained('models/v3/nllb-rus-man-14000')
    fix_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained('models/v3/nllb-rus-man-14000')
    
    df_train = pd.read_csv("trainBig.csv")
    
    # Set up model and move to GPU
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    model.train()
    x, y, loss = None, None, None

    batch_size = 8 # 32 already doesn't fit well to 15GB of GPU memory
    max_length = 1024
    warmup_steps = 1_000 // 8
    training_steps = 57000 // 4

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=4e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )

    losses = []
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    for i in trange(len(losses), training_steps, disable=rank not in [-1, 0], desc="Step: "):
        xx, yy, lang1, lang2 = get_batch_pairs_vocab(batch_size, df_train)

        # Tokenize input
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(rank)
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(rank)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss

        # Forward pass and backward pass
        loss = model(**x, labels=y.input_ids).loss
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Reporting loss
        if i % 125 == 0 and rank in [-1, 0]:
            # Each 1000 steps, report average loss at these steps
            print(f"Step {i}, Loss: {np.mean(losses[-125:])}")

        # Save the model periodically
        if i % 1000 == 0 and i > 0 and rank in [-1, 0]:  # Only save from the main process
            model.module.save_pretrained(f'models/v3/nllb-rus-man-{i}')  # Use model.module to save the underlying model
            tokenizer.save_pretrained(f'models/v3/nllb-rus-man-{i}')
            
    if rank in [-1, 0]:
        with open(f'losses_logs_{rank}.pkl', 'wb') as f:
            pickle.dump(losses, f)
    cleanup()

# Main function to launch the training on multiple GPUs
if __name__ == '__main__':
    train()