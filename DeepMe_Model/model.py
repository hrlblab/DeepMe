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

        # Input embedding layer for multivariate features
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )

        # Time position embedding (captures periodicity)
        self.time_embedding = nn.Embedding(seq_length, hidden_size)

        # Regression output layer
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_length, num_features]
        batch_size, seq_length, num_features = x.shape

        # Feature embedding
        feature_emb = self.feature_embedding(x)

        # Add time position encoding
        position_ids = torch.arange(seq_length, device=x.device).expand(batch_size, seq_length)
        time_emb = self.time_embedding(position_ids)
        embeddings = feature_emb + time_emb

        # Create attention mask (ignore all-zero time steps)
        # We consider a timestep valid if it has at least one non-zero feature
        attention_mask = (x.abs().sum(dim=-1) > 1e-6).float()

        # BERT processing
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        # Use [CLS] token output from last hidden layer
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Regression prediction
        return self.regressor(cls_output).squeeze(-1)

    @staticmethod
    def create_sequences(features, targets, seq_length=168):
        """
        Create 7-day sequences and corresponding labels
        Ensure labels are the first hour after the sequence ends
        Skip samples where the label is zero (missing)
        """
        sequences = []
        labels = []

        # Start from the first complete sequence
        for i in range(seq_length, len(features)):
            # Skip if label is zero (missing value)
            if targets[i] == 0:
                continue

            # Take previous 7 days data (168 hours)
            seq = features[i - seq_length:i]
            # Label is the first hour after sequence ends
            label = targets[i]

            sequences.append(seq)
            labels.append(label)

        return np.array(sequences), np.array(labels)