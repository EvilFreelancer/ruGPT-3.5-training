from llm_rs.convert import AutoConverter
from llm_rs import AutoQuantizer, QuantizationType, ContainerType
from pathlib import Path

content_dir = Path('.').resolve()
input_dir = content_dir / 'ruGPT-3.5-13B-lora'
output_dir = content_dir / 'output_ggml'

# Convert the model to fp16 format
converted_model = AutoConverter.convert(input_dir, output_dir)

# Quantize the model to different formats
AutoQuantizer.quantize(converted_model, quantization=QuantizationType.Q4_0, container=ContainerType.GGML)
AutoQuantizer.quantize(converted_model, quantization=QuantizationType.Q4_1, container=ContainerType.GGML)
AutoQuantizer.quantize(converted_model, quantization=QuantizationType.Q5_0, container=ContainerType.GGML)
AutoQuantizer.quantize(converted_model, quantization=QuantizationType.Q5_1, container=ContainerType.GGML)
AutoQuantizer.quantize(converted_model, quantization=QuantizationType.Q8_0, container=ContainerType.GGML)
