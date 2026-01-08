import json

from functools import partial

from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer, SFTConfig
from datasets import Dataset, NamedSplit

import pandas as pd

from tqdm import tqdm

from data import _raw_data,  _preprocess, COLUMNS
from models.structure_to_standard import convert, ANSWER_DEFAULT
from structure import ThreadReasonings, from_answers_and_labels

import outlines


MAX_SEQ_LENGTH = 4096   # Choose any! We auto support RoPE Scaling internally!
DTYPE = None            # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

MODEL_PATH = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
LOAD_IN_4BIT = True     # Use 4bit quantization to reduce memory usage.
MODEL_NAME = 'gptoss'
FULL_TRAIN = False

# MODEL_PATH = "MasterControlAIML/DeepSeek-R1-Qwen-2.5-1.5b-Latest-Unstructured-To-Structured"
# LOAD_IN_4BIT = False
# MODEL_NAME = 'qwen'
# FULL_TRAIN = False

# MODEL_PATH = "google/gemma-3-270m-it"
# LOAD_IN_4BIT = False
# MODEL_NAME = 'gemma'
# FULL_TRAIN = True

MAX_COMMENTS = int(10e9)
# MAX_COMMENTS = int(1000)

# LOAD DATA
DATA_DIR = {
    'test': '../data/temporal/preprocessed_test.pkl',
    'train': '../data/temporal/preprocessed_train.pkl',
}


def load_model(path=MODEL_PATH, **kwargs):
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        device_map='cuda:0',
        full_finetuning=FULL_TRAIN,
        # fast_inference=True
        **kwargs
    )
    return base_model, tokenizer


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


SCHEMA = ThreadReasonings.model_json_schema()

SYSTEM_PROMPT = """
### Role:
You are an expert on toxic language, specializing in annotating the explicit or implicit toxicity of messages from social media.

### Blank JSON Schema:
{SCHEMA}
"""

FULL_PROMPT = SYSTEM_PROMPT + """

### DATA INPUT:
{SAMPLE}

### TASK REQUIREMENT:
Analyze the given text and fill out the fields of the provided JSON Schema.

### RESPONSE:
{RESPONSE}
"""


def formatting_prompts_func(example, eos_token=None):
    example_dict = example.to_dict()

    response_obj = from_answers_and_labels(example_dict)
    response_str = response_obj.json()

    return {
        'st_id': example_dict['st_id'],
        'with_answer': FULL_PROMPT.format(
            SAMPLE=example_dict['text'], SCHEMA=SCHEMA, RESPONSE=response_str
        ) + eos_token,
        'question_only': FULL_PROMPT.format(SAMPLE=example_dict['text'], SCHEMA=SCHEMA, RESPONSE="") + eos_token,
    }


def data_gen(data, tokenizer):
    f = partial(formatting_prompts_func, eos_token=tokenizer.eos_token)
    yield from data.apply(f, axis=1)


def predict(model, tokenizer, datasets, by_comment_data, split='test'):
    struct_model = outlines.from_transformers(model, tokenizer)
    gen = outlines.Generator(struct_model, ThreadReasonings)
    # pkv = calc_prefix_cache(inf_model, tokenizer)

    data_by_thread = datasets[split]
    data_by_comment = by_comment_data[split]

    predictions = {}
    for i, sample in tqdm(enumerate(data_by_thread)):
        pred = gen(sample['question_only'])

        # Get relevant comments
        st_id = sample['st_id']
        st_df = data_by_comment.loc[data_by_comment['st_id'] == st_id]
        to_drop = ['id', 'workerid', 'timestamp'] + [k for k in st_df.columns if k.startswith('answer_') or k.startswith('label_')]
        st_df = st_df.drop(columns=to_drop).drop_duplicates()

        try:
            pred = json.loads(pred)
            print(pred)
            pred_standard = convert(pred, st_df)
        except:
            # if something goes wrong, insert empty answers
            pred_standard = {st_nr: ANSWER_DEFAULT for st_nr in st_df['st_nr']}
            print('Something went wrong... Using empty/default answers.')

        predictions[st_id] = pred_standard

    return predictions


def get_data(tokenizer, max_nr_comments=1000, test_only=False):
    by_comment_data = {key: _raw_data(_dir) for key, _dir in DATA_DIR.items()}
    by_comment_data = {
        key: df.sort_values('st_id').tail(max_nr_comments)
        for key, df in by_comment_data.items() if not test_only or key == 'test'
    }

    by_thread_df = {
        key: _create_thread_text(_preprocess(df), tokenizer, "", max_length=MAX_SEQ_LENGTH//2)
        for key, df in by_comment_data.items()
    }
    by_thread_df = {key: df.drop(columns=['ids']) for key, df in by_thread_df.items()}

    datasets = {
        key: Dataset.from_generator(
            partial(data_gen, by_thread_df[key], tokenizer), split=NamedSplit(key)
        ) for key in by_thread_df
    }

    return by_comment_data, by_thread_df, datasets


if __name__ == '__main__':
    base_model, tokenizer = load_model()
    _, _, datasets = get_data(tokenizer, max_nr_comments=MAX_COMMENTS)

    # # dump data for inspection
    # for key, dataset_df in datasets.items():
    #     dataset_df.to_csv(f'/tmp/{key}.csv')

    # -------------
    #  fine-tuning
    # -------------

    if not FULL_TRAIN:
        to_train = FastLanguageModel.get_peft_model(
            base_model,
            r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",     # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=True,   # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    else:
        to_train = base_model

    trainer = SFTTrainer(
        model=to_train,
        train_dataset=datasets['train'],
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            warmup_steps=30,
            num_train_epochs=1,  # Set this for 1 full training run.
            # max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=MODEL_NAME,
            report_to="wandb",  # Use this for WandB etc
            dataset_text_field='with_answer'
        ),
    )

    trainer_stats = trainer.train()
