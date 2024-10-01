import os
import fire
import torch
import logging
import transformers

from tqdm import tqdm
from copy import deepcopy
from peft import PeftModel
from typing import Optional, Dict, Any, Tuple
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers.utils.versions import require_version
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import StaticCache

from utils.chat_template import CHAT_TEMPLATE, ASSISTANT_START
from utils.data_utils import read_jsonl, extract_response, num_lines, append_jsonl
from utils.minerva_utils import process_results

require_version(
    "torch==2.3.1",
    """The behavior of spda might vary if not this version: https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
Comment out this line if you are sure about the version of torch you are using.""",
)

class MultiGPUStaticCache(StaticCache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is to fix an issue with multi-GPU usage
        # See: https://github.com/huggingface/transformers/pull/32543/commits/92e27d4dfd1e31c1ec5dac7d46d59b754817ec36
        cache_position = cache_kwargs.get("cache_position")
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device=key_states.device)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device=value_states.device)
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out

# if multi-gpu, use MultiGPUStaticCache
if torch.cuda.device_count() > 1:
    transformers.generation.utils.NEED_SETUP_CACHE_CLASSES_MAPPING["static"] = MultiGPUStaticCache

def extract_math(gens):
    results = []
    for g in gens:
        res = process_results(g["response"], g["output"])
        results.append(res)
    em = sum(results) / len(results)
    return em

def eval(
    # required
    base_model: str = None,
    chat_template_name: str = None,
    output_dir: str = None,
    data_file: str = None,

    # model
    peft_model: str = None,
    bf16: bool = False,
    fp16: bool = False,

    # gen
    batch_size: int = 8,
):
    # Path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generation_file = os.path.join(output_dir, "generation.jsonl")
    result_file = os.path.join(output_dir, "result.log")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, verbose=False, padding_side="left")
    tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    # Set pad_token to eos_token
    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        logging.warning("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            model.config.pad_token_id = model.config.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        except AttributeError:
            model.config.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id

    # PEFT
    if peft_model is not None:
        model = PeftModel.from_pretrained(model, peft_model)

    # eval mode
    model.eval()

    # make model faster
    if getattr(model, "_supports_static_cache", False):
        model = torch.compile(model)
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # Load data
    test_data = []
    for item in read_jsonl(data_file):
        assert item["input"] == ""
        item["prompt"] = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": item["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        item["prompt_length"] = len(tokenizer(item["prompt"], return_tensors="pt")["input_ids"][0])
        test_data.append(item)
    num_samples = len(test_data) 

    # Sort by length
    test_data = sorted(test_data, key=lambda x: (x["prompt_length"], x["prompt"]), reverse=True)

    # Load generations
    num_done = num_lines(generation_file)
    num_left = num_samples - num_done
    test_data_remaining = test_data[num_done:]
    print(f"Total: {num_samples}, Done: {num_done}, Left: {num_left}")

    # batchify
    batched_test_data_remaining = []
    for i in range(0, num_left, batch_size):
        batched_test_data_remaining.append(test_data_remaining[i: i + batch_size])

    # generate
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION), tqdm(total=num_samples) as pbar:
        pbar.update(num_done)
        for batch in batched_test_data_remaining:
            prompts = [item["prompt"] for item in batch]

            # tokenize
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to("cuda")

            # gen
            outputs = tokenizer.batch_decode(
                model.generate(
                    **tokenized,
                    max_new_tokens=2048,
                    do_sample=False,
                    use_cache=True,
                    cache_implementation="static" if getattr(model, "_supports_static_cache", False) else None,
                ),
                skip_special_tokens=True
            )
            responses = [extract_response(o, assistant_start=ASSISTANT_START[chat_template_name]) for o in outputs]

            # save
            for id, item in enumerate(batch):
                response = responses[id]
                item_copy = deepcopy(item)
                item_copy["response"] = response
                append_jsonl(generation_file, item_copy)

            # update pbar
            pbar.update(len(batch))

    # compute metrics
    generations = read_jsonl(generation_file)
    assert len(generations) == num_samples, f"Not all samples are generated. {len(generations)} != {num_samples}"
    all_types = sorted(list(set([g["type"] for g in generations])))
    all_levels = sorted(list(set([g["level"] for g in generations])))

    with open(result_file, "w") as f:
        for t in all_types:
            gens = [g for g in generations if g["type"] == t]
            f.write(f"{t}: {extract_math(gens) * 100:.1f}\n")
        for l in all_levels:
            gens = [g for g in generations if g["level"] == l]
            f.write(f"{l}: {extract_math(gens) * 100:.1f}\n")
        f.write(f"Overall: {extract_math(generations) * 100:.1f}\n")

if __name__ == "__main__":
    fire.Fire(eval)