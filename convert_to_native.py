import fire
import torch
from peft import PeftModel, PeftConfig
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm


def translate_state_dict_key(k):  # noqa: C901
    if 'lora' in k:
        return None
    else:
        return k


def convert_to_native(
        model_name: str,
        output_path: str,
        device: str = "cpu",
        enable_offloading: bool = False
):
    assert output_path.endswith(".bin")

    config = PeftConfig.from_pretrained(model_name)
    base_model_path = config.base_model_name_or_path

    base_model = GPT2LMHeadModel.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        device_map={'': device},
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        model_name,
        device_map={'': device},
        torch_dtype=torch.float32,
    )

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()
    del lora_model, base_model
    total = len(lora_model_sd)
    with tqdm(list(lora_model_sd.keys())) as progress_bar:
        for i, k in enumerate(progress_bar):
            # new_k = k
            new_k = translate_state_dict_key(k)
            if new_k is None:
                continue
            v = lora_model_sd.pop(k)
            lora_model_sd[new_k] = v

            if enable_offloading and i <= total // 2:
                # offload half of all tensors to RAM
                lora_model_sd[new_k] = lora_model_sd[new_k].cpu()

    print('Saving state_dict...')
    torch.save(lora_model_sd, f'{output_path}')


if __name__ == '__main__':
    fire.Fire(convert_to_native)
