import subprocess
from pathlib import Path

# Set up paths
content_dir = Path('.').resolve()
train_full_path = content_dir / 'train_full.jsonl'
val_full_path = content_dir / 'val_full.jsonl'

# Run create_chat_set script from rulm
module_directory = Path('rulm/self_instruct').resolve()
subprocess.run(
    ['python', '-m', 'src.data_processing.create_chat_set', str(train_full_path), str(val_full_path)],
    cwd=module_directory,
    check=True
)

# Check if train_full.jsonl exists
if not train_full_path.exists():
    raise FileNotFoundError(f"{train_full_path} does not exist")

# Set size limits
train_size_limit = 400
val_size_limit = 200

# Create limited-size versions of train_full.jsonl and val_full.jsonl
with open(train_full_path, 'r') as train_full, open(content_dir / 'train.jsonl', 'w') as train_limit:
    for _ in range(train_size_limit):
        train_limit.write(next(train_full))

with open(val_full_path, 'r') as val_full, open(content_dir / 'val.jsonl', 'w') as val_limit:
    for _ in range(val_size_limit):
        val_limit.write(next(val_full))
