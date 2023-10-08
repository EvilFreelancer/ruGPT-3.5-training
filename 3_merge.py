from pathlib import Path
from convert_to_native import convert_to_native

content_dir = Path('.').resolve()
output_dir = content_dir / 'output'
merged_path = output_dir / 'pytorch_model.bin'

convert_to_native(
    model_name=str(output_dir),
    output_path=str(merged_path),
    device='cpu',
    enable_offloading=True
)

assert merged_path.exists()
