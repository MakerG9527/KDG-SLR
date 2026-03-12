import torch
import torch.nn as nn
from transformers import AutoTokenizer, MambaModel
import os

class MambaTextEncoder(nn.Module):
    def __init__(self, model_path, output_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mamba_model = MambaModel.from_pretrained(model_path)
        mamba_hidden_dim = self.mamba_model.config.hidden_size
        self.projection = nn.Linear(mamba_hidden_dim, output_dim)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.mamba_model.device) for k, v in inputs.items()}
        outputs = self.mamba_model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        def mean_pooling(hidden_states, attention_mask):
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        sentence_embeddings = mean_pooling(last_hidden_states, inputs['attention_mask'])
        projected_embeddings = self.projection(sentence_embeddings)
        return projected_embeddings