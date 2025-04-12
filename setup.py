r"""This is a file to prepare the model weights under the local dir."""
import torch
from transformers import AutoModelForCausalLM


models = ["m-a-p/YuE-s1-7B-anneal-en-cot", "m-a-p/YuE-s2-1B-general"]
    
for mdl in models:
    model = AutoModelForCausalLM.from_pretrained(
        mdl, 
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        )
    print(f"Download {mdl} model successfully.")
    del model
    torch.cuda.empty_cache()