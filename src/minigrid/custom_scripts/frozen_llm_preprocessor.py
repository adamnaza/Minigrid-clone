import torch_ac
from transformers import AutoTokenizer

from rl_starter_files.utils.format import preprocess_images


def get_obss_preprocessor(obs_space, model_name="google/flan-t5-base", revision="main"):
    # load flan tokenizer
    # converts instruction text to flan token ids
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    # model inputs
    model_obs_space = {"image": obs_space.spaces["image"].shape,
                       "text": len(tokenizer)}

    def preprocess_obss(obss, device=None):
        # extract instructions and tokenize batch
        tokens = tokenizer([obs["mission"] for obs in obss],
                           padding=True, return_tensors="pt")
        # Nested DictLists preserve token IDs and masks when PPO slices batches.
        # packages the observation image and instruction text
        return torch_ac.DictList({
            "image": preprocess_images([obs["image"] for obs in obss], device=device),
            "text": torch_ac.DictList({
                "input_ids": tokens["input_ids"].to(device),
                "attention_mask": tokens["attention_mask"].to(device),
            }),
        })

    return model_obs_space, preprocess_obss
