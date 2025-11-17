import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import numpy as np


class TimeSeriesBERT(nn.Module):
    def __init__(self, num_features, seq_length=168, hidden_size=192, num_layers=6):
        """
        num_features: Number of input features
        seq_length: Sequence length (168 hours = 7 days)
        hidden_size: BERT hidden layer size
        num_layers: Number of BERT layers
        """
        super().__init__()
        self.seq_length = seq_length

        # BERT configuration optimized for time series
        config = BertConfig(
            vocab_size=1,  # Dummy vocabulary size
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            intermediate_size=512,
            max_position_embeddings=seq_length,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.15,
            output_hidden_states=True
        )

        self.bert = BertModel(config)


        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )


        self.time_embedding = nn.Embedding(seq_length, hidden_size)


        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):

        batch_size, seq_length, num_features = x.shape

        feature_emb = self.feature_embedding(x)

        position_ids = torch.arange(seq_length, device=x.device).expand(batch_size, seq_length)
        time_emb = self.time_embedding(position_ids)
        embeddings = feature_emb + time_emb


        attention_mask = (x.abs().sum(dim=-1) > 1e-6).float()


        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )


        cls_output = outputs.last_hidden_state[:, 0, :]

        return self.regressor(cls_output).squeeze(-1)

    @staticmethod
    def create_sequences(features, targets, seq_length=168):

        sequences = []
        labels = []


        for i in range(seq_length, len(features)):
            if targets[i] == 0:
                continue

            seq = features[i - seq_length:i]

            label = targets[i]

            sequences.append(seq)
            labels.append(label)

        return np.array(sequences), np.array(labels)