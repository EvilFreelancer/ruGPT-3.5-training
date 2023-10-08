from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-8bit', device="cuda:0", use_triton=False, quantize_config=None)
tokenizer = AutoTokenizer.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-8bit')

request = "Человек: Сколько весит жираф? Помощник: "
encoded_input = tokenizer(request, return_tensors='pt', add_special_tokens=False).to('cuda')
output = model.generate(
    **encoded_input,
    num_beams=2,
    do_sample=True,
    max_new_tokens=100
)
print(tokenizer.decode(output[0], skip_special_tokens=True))