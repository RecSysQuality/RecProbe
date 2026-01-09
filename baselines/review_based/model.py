import torch
from torch import nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, config, word_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),   # ← FIX CRITICO
            nn.Dropout(p=config.dropout_prob)
        )

        self.linear = nn.Sequential(
            nn.Linear(config.kernel_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B*R, D, L)
        x = self.conv(x)         # (B*R, K, 1)
        x = x.squeeze(-1)        # (B*R, K)
        x = self.linear(x)       # (B*R, cnn_out_dim)
        return x

# class CNN(nn.Module):
#     def __init__(self, config, word_dim):
#         super().__init__()
#         self.kernel_count = config.kernel_count
#         self.review_count = config.review_count
#
#         self.conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=word_dim,
#                 out_channels=config.kernel_count,
#                 kernel_size=config.kernel_size,
#                 padding=(config.kernel_size - 1) // 2  # mantiene la lunghezza
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=config.review_length),  # max pooling lungo la sequenza
#             nn.Dropout(p=config.dropout_prob)
#         )
#
#         self.linear = nn.Sequential(
#             nn.Linear(config.kernel_count, config.cnn_out_dim),  # ora conv riduce a kernel_count
#             nn.ReLU(),
#             nn.Dropout(p=config.dropout_prob)
#         )
#
#     def forward(self, x):
#         # x: (batch*review_count, review_length, word_dim)
#         x = x.permute(0, 2, 1)  # (batch*review_count, word_dim, review_length)
#         x = self.conv(x)        # (batch*review_count, kernel_count, 1)
#         x = x.squeeze(-1)       # (batch*review_count, kernel_count)
#         x = self.linear(x)      # (batch*review_count, cnn_out_dim)
#         return x


class FactorizationMachine(nn.Module):
    def __init__(self, p, k):  # p = cnn_out_dim*2
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output


class DeepCoNN(nn.Module):
    def __init__(self, config, word_emb):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb), freeze=False)
        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, user_review, item_review):
        """
        user_review: (batch_size, review_count, review_length)
        item_review: (batch_size, review_count, review_length)
        """
        batch_size, review_count, review_length = user_review.shape

        # Embedding
        u_vec = self.embedding(user_review.reshape(-1, review_length))  # (batch*review_count, review_length, emb_dim)
        i_vec = self.embedding(item_review.reshape(-1, review_length))

        # CNN
        user_latent = self.cnn_u(u_vec)  # (batch*review_count, cnn_out_dim)
        item_latent = self.cnn_i(i_vec)
        #user_latent = F.normalize(user_latent, p=2, dim=1)
        #item_latent = F.normalize(item_latent, p=2, dim=1)
        # Media sulle review per ottenere un vettore per utente/item
        user_latent = user_latent.view(batch_size, review_count, -1).mean(dim=1)
        item_latent = item_latent.view(batch_size, review_count, -1).mean(dim=1)

        # Concatenazione e FM
        concat_latent = torch.cat((user_latent, item_latent), dim=1)  # (batch_size, cnn_out_dim*2)
        prediction = self.fm(concat_latent)  # (batch_size, 1)
        return prediction
