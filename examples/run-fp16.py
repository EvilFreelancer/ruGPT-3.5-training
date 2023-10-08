from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-fp16')
tokenizer = AutoTokenizer.from_pretrained('Gaivoronsky/ruGPT-3.5-13B-fp16')
model = model.half()
model = model.to('cuda')

request = "Человек: Сколько весит жираф? Помощник: "
encoded_input = tokenizer(request, return_tensors='pt', \
                          add_special_tokens=False).to('cuda')
output = model.generate(
    **encoded_input,
    num_beams=2,
    do_sample=True,
    max_new_tokens=100
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
