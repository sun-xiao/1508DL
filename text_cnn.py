import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)      # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)     # (batch_size, embed_dim, seq_len)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [
            F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            for conv_out in conv_outputs
        ]

        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x).squeeze(1)  # (batch_size,)

        return logits