from functools import partial

from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

import pandas as pd

from tqdm import tqdm

from data import _raw_data,  _preprocess, COLUMNS
from structure import ThreadReasonings, from_answers_and_labels


MAX_SEQ_LENGTH = 4096   # Choose any! We auto support RoPE Scaling internally!
DTYPE = None            # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True     # Use 4bit quantization to reduce memory usage. Can be False.

MODEL_PATH = "MasterControlAIML/DeepSeek-R1-Qwen-2.5-1.5b-Latest-Unstructured-To-Structured"


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0, # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)


# LOAD DATA
DATA_DIR = {
    'test': '../data/temporal/preprocessed_test.pkl',
    'train': '../data/temporal/preprocessed_train.pkl',
}
eos_token = tokenizer.eos_token


def _create_thread_text(comments_df, tokenizer, comment_token, add_post_text=False, max_length=500):
    nr_long_msgs = 0
    nr_skip_msgs = 0

    result = []
    for (workerid, st_id), st_df in tqdm(comments_df.groupby(by=['workerid', 'st_id'])):
        row = {'st_id': st_id}

        title = st_df['subm_title'].unique().tolist()[0]
        post = st_df['subm_body'].unique().tolist()[0]
        subreddit = st_df['subreddit'].unique().tolist()[0]

        post_title = '`' + title + '`' if not pd.isna(title) else 'EMPTY'
        if add_post_text:
            post_text = '\n```\n' + post + '\n```' if not pd.isna(post) else 'EMPTY'
        else:
            post_text = 'HIDDEN'

        start_str = f"""\
From a thread in r/{subreddit}

Post Title: {post_title}
Post Text: {post_text}

"""
        st_df = st_df.sort_values('st_nr')

        message_counts = [0]
        message_strs = [""]
        message_ids = [[]]
        cols = ['st_nr', 'comment_id', 'author_name', 'comment_body']
        for i, (st_nr, comment_id, author, comment_body) in enumerate(st_df[cols].itertuples(index=False)):
            msg = comment_token + f"Message {i + 1} (by {author}):\n```\n{comment_body}\n```\n\n"

            if len(tokenizer.encode(start_str + message_strs[-1] + msg)) > max_length:
                # adding the current comment would make the existing message too long, add a new empty message
                message_strs.append("")
                message_counts.append(0)
                message_ids.append([])

            message_strs[-1] += msg
            message_counts[-1] += 1
            message_ids[-1].append((st_nr, comment_id))

        if len(message_strs) > 1:
            nr_long_msgs += 1

        new_rows = []
        for i, msg_str in enumerate(message_strs):
            if message_counts[i] == 0:
                continue

            new_row = row.copy()
            new_row['text'] = start_str + msg_str
            new_row['ids'] = message_ids[i]

            if len(tokenizer.encode(new_row['text'])) > max_length:
                nr_skip_msgs += 1
                continue

            start = sum(message_counts[:i])
            end = start + message_counts[i]

            for key in [k for k in st_df.columns if k.startswith('answer_') or k.startswith('label_')]:
                by_comment = st_df.iloc[start:end][key].tolist()
                new_row[key] = by_comment

            new_rows.append(new_row)
        result.extend(new_rows)

    print(f'Split up {nr_long_msgs} that were too long otherwise.')
    print(f'Skipped {nr_skip_msgs} that were still too long after.')
    return pd.DataFrame(result)


by_comment_data = {key: _raw_data(_dir) for key, _dir in DATA_DIR.items()}
by_comment_data = {key: df.sort_values('st_id').iloc[-1000:] for key, df in by_comment_data.items()}

by_thread_df = {
    key: _create_thread_text(_preprocess(df), tokenizer, "", max_length=MAX_SEQ_LENGTH/2)
    for key, df in by_comment_data.items()
}
by_thread_df = {key: df.drop(columns=['ids']) for key, df in by_thread_df.items()}



SYSTEM_PROMPT = """
### Role:
You are an expert on toxic language, specializing in annotating the explicit or implicit toxicity of messages from social media.

### DATA INPUT:
- **Messages:** ```{SAMPLE}```  
- **Blank JSON Schema:** ```{SCHEMA}```  

### TASK REQUIREMENT:
Analyze the given text and fill out the fields of the provided JSON Schema.

### RESPONSE:
{RESPONSE}
"""

schema = ThreadReasonings.model_json_schema()


def formatting_prompts_func(example):
    example_dict = example.to_dict()

    response_obj = from_answers_and_labels(example_dict)
    response_str = response_obj.json()
    text = SYSTEM_PROMPT.format(
        SAMPLE=example_dict['text'],
        SCHEMA=schema,
        RESPONSE=response_str
    ) + eos_token

    return {'text': text}


def gen(key):
    yield from by_thread_df[key].apply(formatting_prompts_func, axis=1)


datasets = {key: Dataset.from_generator(partial(gen, key)) for key in by_thread_df}

# datasets = {key:  for key, dataset in by_thread_df.items()}
# datasets = {key: dataset.map(formatting_prompts_func, batched=True) for key, dataset in datasets.items()}

#
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=datasets["train"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False, # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()
