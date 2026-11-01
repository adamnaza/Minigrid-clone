import torch
from torch import nn
from transformers import T5EncoderModel

from rl_starter_files.model import ACModel, init_params


class FrozenLLMACModel(ACModel):
    # initialize default flan-t5-base LLM
    def __init__(self, obs_space, action_space, use_memory=False,
                 model_name="google/flan-t5-base", revision="main"):

        # initialize original rl-starter-file learning agent
        # Reuse the CNN, actor, critic, and forward method unchanged. Initialize
        # these before attaching FLAN so init_params cannot overwrite its weights.
        super().__init__(obs_space, action_space, use_memory, use_text=True)
        
        # remove original agent's text encoder
        del self.word_embedding
        del self.text_rnn

        # load flan encoder
        self.encoder = T5EncoderModel.from_pretrained(model_name, revision=revision)
        # freeze flan encoder and set to eval mode
        self.encoder.float().requires_grad_(False)
        self.encoder.eval()

        # projection head layer to map 768 dimension vector from flan encoder to 128 for baby ai learning agent
        self.projection = nn.Linear(self.encoder.config.d_model, self.text_embedding_size)
        self.projection.apply(init_params)

    def train(self, mode=True):
        # train agent with frozen llm weights
        super().train(mode)
        # PPO calls model.train(); the frozen encoder must keep dropout disabled.
        self.encoder.eval()
        return self

    def parameters(self, recurse=True):
        # torch-ac computes gradient norms without checking for grad=None.
        # Expose only trainable parameters to its optimizer and gradient logging.
        # named_parameters/state_dict still include the full frozen encoder.
        return (p for p in super().parameters(recurse=recurse) if p.requires_grad)

    def _get_embed_text(self, text):
        # get output from frozen encoder
        with torch.no_grad():
            hidden = self.encoder(
                input_ids=text.input_ids, attention_mask=text.attention_mask
            ).last_hidden_state
            mask = text.attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        # This layer learns through PPO, so it stays outside no_grad().
        # send 768 dimension vector to project head to transform to expected 128 dimension vector
        return self.projection(pooled)
