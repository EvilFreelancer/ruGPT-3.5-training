import subprocess
from pathlib import Path
import json

content_dir = Path('.').resolve()
output_dir = content_dir / 'output'
merged_path = output_dir / 'pytorch_model.bin'

# Prepare tokenizer.json
with open(output_dir / 'tokenizer_config.json', 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
with open(output_dir / 'vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
with open(output_dir / 'merges.txt', 'r', encoding='utf-8') as f:
    merges = f.read().strip().split('\n')

tokenizer_config['model'] = {}
tokenizer_config['model']['vocab'] = vocab
tokenizer_config['model']['merges'] = merges
with open(output_dir / 'tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

# Convert to GGML
ggml_f16_model_name = 'ggml-model-f16.bin'

module_directory = Path('llama.cpp').resolve()
# subprocess.run(
#     [
#         'python', 'convert-gpt2-hf-to-gguf.py',
#         '--outfile', str(output_dir / ggml_f16_model_name),
#         str(output_dir),
#         str(1),  # output format - use 0 for float32, 1 for float16
#     ],
#     cwd=module_directory,
#     check=True
# )
#
# assert (output_dir / ggml_f16_model_name).exists()

# Quantize to 4bit
quantization_type = "q4_0"
ggml_quantized_model_name = f'ggml-model-{quantization_type}.bin'

subprocess.run(
    ['make', 'quantize'],
    cwd=module_directory,
    check=True
)
subprocess.run(
    [
        './quantize',
        str(output_dir / ggml_f16_model_name),
        str(output_dir / ggml_quantized_model_name),
        quantization_type
    ],
    cwd=module_directory,
    check=True
)
# module_directory = Path('ggml').resolve()
# subprocess.run(
#     ['cmake', '.'],
#     cwd=module_directory,
#     check=True
# )
# subprocess.run(
#     ['make', 'gpt-2', 'gpt-2-quantize'],
#     cwd=module_directory,
#     check=True
# )
# subprocess.run(
#     [
#         './bin/gpt-2-quantize',
#         str(output_dir / ggml_f16_model_name),
#         str(output_dir / ggml_quantized_model_name),
#         quantization_type
#     ],
#     cwd=module_directory,
#     check=True
# )

assert (output_dir / ggml_quantized_model_name).exists()
