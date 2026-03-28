import torch
import torch.nn as nn


class TextLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * num_directions

        # Attention layer: learns which timesteps matter most
        self.attn_fc = nn.Linear(lstm_output_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)                  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)             # (batch_size, seq_len, hidden_dim * num_directions)

        # Attention: score each timestep, then weighted sum
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)   # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (batch_size, seq_len)
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_out
        ).squeeze(1)                                        # (batch_size, hidden_dim * num_directions)

        context = self.dropout(context)
        logits = self.fc(context).squeeze(1)   # (batch_size,)

        return logits
