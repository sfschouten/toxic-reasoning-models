import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel, get_scheduler
from accelerate.test_utils.testing import get_backend

import wandb
from tqdm import tqdm

from data import comtok_load_data

# from model_xlm_roberta import XLMRobertaForToxicReasoning
from model_eurobert import EuroBertForToxicReasoning


NUM_EPOCHS = 6
MAX_NO_IMPROVEMENT = 999  # disabled
FRAC_WARMUP_STEPS = 0.1
DATA_DIR = {
    # 'eval': '../data/temporal/raw_eval_data/all/',
    'test': '../data/temporal/preprocessed_test.pkl',
    # 'train': '../git-repo/local_outputs/gpt4o_combined/',
    'train': '../data/temporal/preprocessed_train.pkl',
}
# WEIGHT_KEY = "FacebookAI/xlm-roberta-base"
# WEIGHT_KEY = "FacebookAI/xlm-roberta-large"
WEIGHT_KEY = "EuroBERT/EuroBERT-210m"
# WEIGHT_KEY = "EuroBERT/EuroBERT-610m"
# WEIGHT_KEY = "EuroBERT/EuroBERT-2.1B"

# MAX_LENGTH = 500
MAX_LENGTH = 1000

SIZE = WEIGHT_KEY.split("-")[-1]

MODEL_NAME = 'eurobert'
# MODEL_NAME = 'xlm-roberta'
CACHE_DIR = f'cache-{MODEL_NAME}-{SIZE}-{MAX_LENGTH}/'

DTYPE = torch.bfloat16

CLASSIFIER_DROPOUT = 0.1

# BATCH_SIZE = 64
# BATCH_SIZE = 21
# BATCH_SIZE = 16
BATCH_SIZE = 32
ACC_STEPS = 2
# ACC_STEPS = 1

BASE_LR = 3e-5
HEADS_LR = 5e-3

BASE_WD = 0.01
HEADS_WD = 0


COMMENT_TOKEN = "<COMMENT>"

device, _, _ = get_backend()  # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)

# Instantiate model
tokenizer = AutoTokenizer.from_pretrained(WEIGHT_KEY)
tokenizer.add_special_tokens({'additional_special_tokens': [COMMENT_TOKEN]})

# model = XLMRobertaForToxicReasoning.from_pretrained(
model = EuroBertForToxicReasoning.from_pretrained(
    WEIGHT_KEY, device_map=device, torch_dtype=DTYPE, classifier_dropout=CLASSIFIER_DROPOUT, use_flash_attention_2=True
)
model.further_init(tokenizer.vocab[COMMENT_TOKEN])

# Load data
train_dataloader, dev_dataloader, test_dataloader = comtok_load_data(
    DATA_DIR['train'], DATA_DIR['test'], CACHE_DIR, tokenizer, COMMENT_TOKEN, BATCH_SIZE, max_length=MAX_LENGTH
)

# input_lengths = [sum(x) for x in train_dataloader.dataset['attention_mask']]
# print(np.histogram(input_lengths))

# Training loop

optimizer = AdamW([
    {'params': model.model.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},
    {'params': model.cls.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
])

num_training_steps = NUM_EPOCHS * len(train_dataloader) // ACC_STEPS
num_warmup_steps = int(num_training_steps * FRAC_WARMUP_STEPS)
lr_scheduler = get_scheduler(
    name='linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


run = wandb.init(
    project="xlm-eurobert-toxic-reasoning",
    config={
        "weights": WEIGHT_KEY,
        "base-lr": BASE_LR, "heads-lr": HEADS_LR, "epochs": NUM_EPOCHS, "classifier_dropout": CLASSIFIER_DROPOUT,
        'batch_size': BATCH_SIZE, 'acc_steps': ACC_STEPS,
    },
)


print(f'Nr. of trainable parameters: {model.num_parameters(only_trainable=True)}')

best_epoch_loss = float('inf')
last_improvement = -1

for epoch in range(NUM_EPOCHS):
    model.train()

    pbar = tqdm(train_dataloader)
    acc_cnt = 0
    for batch in pbar:
        inputs = {k: v.to(device) for k, v in batch.items() if type(v) is torch.Tensor}
        _, losses = model(**inputs)
        loss = losses['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        acc_cnt += 1

        pbar.set_postfix(loss=loss.item())
        run.log({f"loss_{key}": value.item() for key, value in losses.items()})
        if acc_cnt % ACC_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    model.eval()
    total_dev_loss = 0
    for batch in dev_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if type(v) is torch.Tensor}
        with torch.no_grad():
            _, losses = model(**inputs)

        if not torch.isnan(losses['total']):
            total_dev_loss += losses['total'].item()
        else:
            print('The loss of one batch is NaN.')

    if total_dev_loss < best_epoch_loss:  # we improved
        best_epoch_loss = total_dev_loss
        model.save_pretrained(f'./saved_{MODEL_NAME}_{SIZE}/')
        last_improvement = epoch
    elif last_improvement + MAX_NO_IMPROVEMENT == epoch:
        print(f'No improvement for {MAX_NO_IMPROVEMENT} epochs, calling it.')
        break

    run.log({f"dev_loss_total": total_dev_loss / len(dev_dataloader)})


