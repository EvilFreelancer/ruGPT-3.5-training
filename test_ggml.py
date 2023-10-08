from pathlib import Path
import json
from llm_rs import AutoModel, GenerationConfig

content_dir = Path('.').resolve()
output_dir = content_dir / 'output'
output_ggml_dir = content_dir / 'output_ggml'
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — ruGPT-3.5, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            start_token_id=2,
            bot_token_id=46787
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, prompt, generation_config):
    data = model.tokenize(prompt)
    output = model.generate(
        prompt=prompt,
        generation_config=generation_config
    ).text
    output_ids = model.tokenize(output)
    output_ids = output_ids[len(data):]
    output = model.decode(output_ids)
    return output.strip()


# Load base model
model = AutoModel.from_pretrained(
    output_ggml_dir,
    model_file="ruGPT-3.5-13B-lora-q4_0.bin",
)

# Instantiate generator config
with open(output_dir / 'generation_config.json', 'r', encoding='utf-8') as f:
    generation_config = json.load(f)

g_config = GenerationConfig()
g_config.top_p = generation_config['top_p']
g_config.top_k = generation_config['top_k']
g_config.repetition_penalty = generation_config['repetition_penalty']
g_config.temperature = generation_config['temperature']
g_config.max_new_tokens = generation_config['max_new_tokens']

# Start conversation
conversation = Conversation()
while True:
    user_message = input("User: ")
    if user_message.strip() == "/reset":
        conversation = Conversation()
        print("History reset completed!")
        continue
    conversation.add_user_message(user_message)
    prompt = conversation.get_prompt(model)
    output = generate(
        model=model,
        prompt=prompt,
        generation_config=g_config
    )
    conversation.add_bot_message(output)
    print("ruGPT-3.5:", output)
    print()
    print("==============================")
    print()
