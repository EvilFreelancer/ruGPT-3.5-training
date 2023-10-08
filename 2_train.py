import json
from huggingface_hub import snapshot_download
from pathlib import Path
import subprocess

content_dir = Path('.').resolve()
original_config_path = content_dir / 'configs/rugpt35_13b.json'
model_dir = content_dir / "ruGPT-3.5-13B"
base_model = "ai-forever/ruGPT-3.5-13B"
output_dir = content_dir / 'output'
config_path = content_dir / 'configs/rugpt35_13b_colab.json'

# Paths to datasets
train_full_path = content_dir / 'train_full.jsonl'
train_small_path = content_dir / 'train.jsonl'
train_path = train_full_path  # change to train_full_path if you need
val_full_path = content_dir / 'val_full.jsonl'
val_small_path = content_dir / 'val.jsonl'
val_path = val_full_path  # change to val_full_path if you need

# Download binaries
snapshot_download(repo_id=base_model, local_dir=model_dir, ignore_patterns=["LICENSE", "README.md", ".gitattributes"])

patch_model_config = True

if patch_model_config:
    replacements = {
        "tokenizer_config.json": {
            "add_bos_token": False,
            "add_prefix_space": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "<s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "clean_up_tokenization_spaces": True,
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "errors": "replace",
            "mask_token": "<mask>",
            "model_max_length": 2048,
            "pad_token": {
                "__type": "AddedToken",
                "content": "<pad>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "tokenizer_class": "GPT2Tokenizer",
            "unk_token": {
                "__type": "AddedToken",
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "padding_side": "left"
        },
        "special_tokens_map.json": {
            "bos_token": {
                "content": "<s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "eos_token": {
                "content": "</s>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "mask_token": "<mask>",
            "pad_token": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            },
            "sep_token": "<s>",
            "unk_token": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False
            }
        },
        "generation_config.json": {
            "_from_model_config": True,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 0,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 30,
            "do_sample": True,
            "max_new_tokens": 1536,
            "num_beams": 1,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 15
        },
    }

    print('Patching model config...')
    for filename, new_content in replacements.items():
        print(f'{filename}:')
        with (model_dir / filename).open() as fp:
            old_content = json.load(fp)
            print(f'    Original content: {old_content}')
            if old_content == new_content:
                print('    Already patched, skipping')
        print(f'    Updated content:  {new_content}')
        with (model_dir / filename).open('w') as fp:
            json.dump(new_content, fp, indent=4)

# Load configurations
with original_config_path.open('r') as fp:
    config = json.load(fp)

# Colab adjustments
config['trainer']['per_device_train_batch_size'] = 2
config['trainer']['per_device_eval_batch_size'] = 1
config['trainer']['gradient_accumulation_steps'] = 128
config['trainer']['eval_steps'] = 50
config['trainer']['save_steps'] = 50
config['max_tokens_count'] = 1000
#config['model_name'] = str(model_dir)
config['templates_path'] = str(content_dir / 'internal_prompts/rugpt35.json')
config['load_in_8bit'] = True
config['load_in_4bit'] = False

# Demo adjustments
config['trainer']['eval_steps'] = 2
config['trainer']['logging_steps'] = 1
config['trainer']['num_train_epochs'] = 1

with config_path.open('w') as fp:
    json.dump(config, fp, indent=4)

# Run training
module_directory = Path('rulm/self_instruct').resolve()
subprocess.run(
    [
        'python', '-m', 'src.train',
        '--config-file', config_path,
        '--train-file', train_path,
        '--val-file', val_path,
        '--output-dir', output_dir,
        '--report-to', 'none'
    ],
    cwd=module_directory,
    check=True
)

assert (output_dir / 'adapter_config.json').exists()

# Fix config of trained model
with (output_dir / 'generation_config.json').open('w') as fp:
    json.dump({
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 30,
        "do_sample": True,
        "max_new_tokens": 1536,
        "num_beams": 1,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 15,
    }, fp, indent=4)
