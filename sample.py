# %%
import torch
from transformers import StoppingCriteriaList, StoppingCriteria
from eindex import eindex
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from transformer_lens import HookedTransformer, utils, HookedTransformerKeyValueCache
from transformer_lens.utils import USE_DEFAULT_VALUE
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Literal
# %%
torch.set_grad_enabled(False)

from typing import List, Optional, Union, Literal
from jaxtyping import Float, Int


from prettytable import PrettyTable
import textwrap

def create_wrapped_table(completions, max_width=80):
    table = PrettyTable()
    table.field_names = ["Completion"]
    table.align["Completion"] = "l"  # Left-align text
    table.max_width = max_width

    wrapper = textwrap.TextWrapper(width=max_width - 4)  # -4 for padding and borders

    for completion in completions:
        wrapped_text = wrapper.fill(completion)
        table.add_row([wrapped_text])
        table.add_row(["====================================="])  # Add an empty row between completions

    return table


@torch.inference_mode()
def generate(
    model: HookedTransformer,
    tokens: Float[torch.Tensor, "batch pos"],
    max_new_tokens: int = 10,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    use_past_kv_cache: bool = True,
    prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
    padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    verbose: bool = True,
) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:

    with utils.LocallyOverridenDefaults(
        model, prepend_bos=prepend_bos, padding_side=padding_side
    ):
        device = next(model.parameters()).device
        log_probs = None
        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape

        tokens = tokens.to(device)
        if use_past_kv_cache:
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                model.cfg, model.cfg.device, batch_size
            )
        else:
            past_kv_cache = None

        stop_tokens: List[int] = []
        eos_token_for_padding = 0
        assert model.tokenizer is not None

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=model.cfg.device)

        # Currently nothing in HookedTransformer changes with eval, but this is here in case
        # that changes in the future.
        model.eval()
        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index > 0:
                    logits = model.forward(
                        tokens[:, -1:],
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )
                else:
                    logits = model.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )
            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                logits = model.forward(
                    tokens,
                    return_type="logits",
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                )
            final_logits = logits[:, -1, :] # [batch, vocab_size]
            
            if do_sample:
                sampled_tokens = utils.sample_logits(
                    final_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                    tokens=tokens,
                )
            else:
                sampled_tokens = final_logits.argmax(-1)

            sampled_log_probs = torch.log_softmax(final_logits, dim=-1) # [batch, vocab_size]
            sampled_log_probs = eindex(sampled_log_probs, sampled_tokens, "batch [batch]") # [batch]

            if log_probs is None:
                log_probs = sampled_log_probs.unsqueeze(-1) # [batch, 1]
            else:
                log_probs = torch.cat([log_probs, sampled_log_probs.unsqueeze(-1)], dim=-1)

            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

        tokens = tokens[:, ctx_length:]
        assert tokens.shape == log_probs.shape, f"{tokens.shape=} != {log_probs.shape=}"
        return tokens, log_probs

# %%
def big_sample(model, prompt, max_new_tokens=10, num_completions = 50, batch_size = 2, **kwargs):

    input_ids = model.to_tokens(prompt)
    all_tokens, all_log_probs = [], []

    num_loops = num_completions // batch_size
    last_batch_size = num_completions % batch_size
    if last_batch_size > 0:
        num_loops += 1

    runner =  tqdm.tqdm(range(num_loops), total = num_completions)
    for i in runner:
        current_batch_size = last_batch_size if i == num_loops - 1 else batch_size
        batch_input_ids = einops.repeat(input_ids, "1 seq -> batch seq", batch=current_batch_size)
        tokens, log_probs = generate(model, batch_input_ids, max_new_tokens=max_new_tokens, **kwargs)
        all_tokens.append(tokens)
        all_log_probs.append(log_probs)
        runner.update(current_batch_size)

        completions = [prompt + model.tokenizer.decode(y) for y in tokens[:3]]
        
        table = create_wrapped_table(completions, max_width=80)
        print(table)
    print("catting results...")
    all_tokens = torch.cat(all_tokens, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)

    return all_tokens, all_log_probs

# %%

# %%
torch.set_float32_matmul_precision('medium')
import argparse 
import pickle 

@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    prompt: str = ""
    do_sample: bool = True
    temperature: float = 1
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    freq_penalty: float = 0
    max_new_tokens: int = 100
    num_completions: int = 10_000
    batch_size: int = 256
    tokens: Optional[Float[torch.Tensor, "batch pos"]] = None
    log_probs: Optional[Float[torch.Tensor, "batch pos_plus_new_tokens"]] = None
    out_file: Optional[str] = 'DUMMY_FILE.pt'
    device : str = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Llama Sampling Script")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for sampling")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--freq_penalty", type=float, default=0, help="Frequency penalty")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_completions", type=int, default=10_000, help="Number of completions to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    parser.add_argument("--out_file", type=str, default='DUMMY_FILE.pt', help="Output file name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(**vars(args))

    torch.manual_seed(42)  # Fix the seed for reproducibility
    device = torch.device(cfg.device)
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.model_name, device=device, dtype=torch.bfloat16)
    #model = torch.compile(model)
    print(asdict(cfg))
    print("Starting sampling...")
    out = big_sample(model, cfg.prompt, 
                     max_new_tokens=cfg.max_new_tokens, 
                     num_completions=cfg.num_completions, 
                     batch_size=cfg.batch_size, 
                     verbose=False)
    print("Finished sampling, processing results...")
    tokens, log_probs = out
    tokens = tokens.cpu().numpy()
    log_probs = log_probs.cpu().float().numpy()
    cfg.tokens = tokens
    cfg.log_probs = log_probs

    print("Saving results...")
    
    with open(cfg.out_file, 'wb') as f:
        pickle.dump(asdict(cfg), f)

    torch.cuda.empty_cache()
    print("Done!")

if __name__ == "__main__":
    main()
# %%
