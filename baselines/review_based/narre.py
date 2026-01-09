# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class NARRE(nn.Module):
    def __init__(self, opt, word_emb=None):
        super(NARRE, self).__init__()
        self.opt = opt

        # Embedding
        self.user_id_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_id_emb = nn.Embedding(opt.item_num, opt.id_emb_size)
        self.user_bias = nn.Embedding(opt.user_num, 1)
        self.item_bias = nn.Embedding(opt.item_num, 1)

        # CNN + linear
        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.fc_layer = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.dropout = nn.Dropout(opt.drop_out)
        self.attention_linear = nn.Linear(opt.id_emb_size, 1)
        self.review_linear = nn.Linear(opt.filters_num, opt.id_emb_size)

        # 🔹 Predizione finale
        self.pred_layer = nn.Linear(2 * opt.id_emb_size, 1)  # basta concatenare user+item

        # Word embedding
        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        if word_emb is not None:
            self.word_embs.weight.data.copy_(word_emb)
            self.word_embs.weight.requires_grad = False  # velocità

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)
        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)
        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)
        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)
        nn.init.uniform_(self.user_id_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_id_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.user_bias.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_bias.weight, -0.1, 0.1)

    def encode_reviews(self, reviews):
        """
        Pre-computa embedding delle review per batch.
        reviews: [batch, review_count, review_length]
        ritorna: [batch, review_count, id_emb_size]
        """
        bs, r_num, r_len = reviews.size()
        x = self.word_embs(reviews)  # [bs, r_num, r_len, word_dim]
        x = x.view(-1, r_len, self.opt.word_dim)  # [bs*r_num, r_len, word_dim]

        # CNN
        fea = F.relu(self.cnn(x.unsqueeze(1))).squeeze(3)  # [bs*r_num, filters_num]
        fea = F.adaptive_max_pool1d(fea, 1).squeeze(2)
        fea = fea.view(bs, r_num, -1)  # [bs, review_count, filters_num]

        # Linear layer
        fea = self.fc_layer(fea)  # [bs, review_count, id_emb_size]
        return fea

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas

        # 1️⃣ User & Item ID embedding
        #print("UID min:", uids.min().item(), "UID max:", uids.max().item())
        #print("Num users:", self.user_id_emb.num_embeddings)
        max_valid_uid = self.user_id_emb.num_embeddings - 1
        uids_safe = torch.clamp(uids, max=max_valid_uid)

        max_valid_iid = self.item_id_emb.num_embeddings - 1
        iids_safe = torch.clamp(iids, max=max_valid_iid)

        # 2️⃣ User & Item ID embedding
        u_id_emb = self.user_id_emb(uids_safe)
        i_id_emb = self.item_id_emb(iids_safe)
        # 2️⃣ Encode review embedding
        u_rev_emb = self.review_attention(self.encode_reviews(user_reviews))
        i_rev_emb = self.review_attention(self.encode_reviews(item_reviews))

        # 3️⃣ Somma ID + review embedding
        u_all = u_id_emb + u_rev_emb
        i_all = i_id_emb + i_rev_emb
        x = torch.cat([u_all, i_all], dim=1)  # [batch, 2*id_emb_size]

        # 4️⃣ Linear prediction
        #rating = self.pred_layer(x)
        user_bias_safe = torch.clamp(uids, max=self.user_bias.num_embeddings - 1)
        item_bias_safe = torch.clamp(iids, max=self.item_bias.num_embeddings - 1)

        #rating = self.pred_layer(x) + self.user_bias(uids) + self.item_bias(iids)
        rating = self.pred_layer(x) + self.user_bias(user_bias_safe) + self.item_bias(item_bias_safe)

        #rating = rating + self.user_bias(uids) + self.item_bias(iids)
        return rating

    def review_attention(self, fea):
        """
        fea: [batch, review_count, id_emb_size]
        """
        att_score = self.attention_linear(fea)  # [bs, review_count, 1]
        att_weight = F.softmax(att_score, dim=1)
        r_fea = (fea * att_weight).sum(dim=1)  # [bs, id_emb_size]
        r_fea = self.dropout(r_fea)
        return r_fea

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class NARRE(nn.Module):
#     def __init__(self, opt, word_emb):
#         super(NARRE, self).__init__()
#         self.opt = opt
#         self.num_fea = 2  # ID + Review
#
#         self.user_net = Net(opt, 'user', word_emb)
#         self.item_net = Net(opt, 'item', word_emb)
#
#         # Prediction layer: concatenazione delle feature + linear layer
#         self.pred_layer = nn.Linear(4 * opt.id_emb_size, 1)
#         self.user_bias = nn.Embedding(opt.user_num, 1)
#         self.item_bias = nn.Embedding(opt.item_num, 1)
#
#     def forward(self, datas):
#         user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
#
#         # embedding utente/item
#         u_fea = self.user_net(user_reviews, uids, user_item2id)
#         i_fea = self.item_net(item_reviews, iids, item_user2id)
#
#         # Prendi ID embedding e review embedding
#         u_id_emb = u_fea[:, 0, :]
#         u_rev_emb = u_fea[:, 1, :]
#         i_id_emb = i_fea[:, 0, :]
#         i_rev_emb = i_fea[:, 1, :]
#
#         # Concatena tutte le feature
#         u_all = torch.cat([u_id_emb, u_rev_emb], dim=1)
#         i_all = torch.cat([i_id_emb, i_rev_emb], dim=1)
#         x = torch.cat([u_all, i_all], dim=1)  # [batch, 4*id_emb_size]
#
#         # Predizione + bias
#         rating = self.pred_layer(x)
#         rating = rating + self.user_bias(uids) + self.item_bias(iids)
#         return rating  # [batch, 1]
#
#
# class Net(nn.Module):
#     def __init__(self, opt, uori='user', word_emb=None):
#         super(Net, self).__init__()
#         self.opt = opt
#
#         if uori == 'user':
#             id_num = self.opt.user_num
#             ui_id_num = self.opt.item_num
#         else:
#             id_num = self.opt.item_num
#             ui_id_num = self.opt.user_num
#
#         self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)
#         self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
#         if word_emb is not None:
#             self.word_embs.weight.data.copy_(word_emb)
#         else:
#             nn.init.xavier_normal_(self.word_embs.weight)
#
#         self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)
#
#         self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
#         self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)
#         self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)
#         self.fc_layer = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
#
#         self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
#         self.dropout = nn.Dropout(self.opt.drop_out)
#         self.reset_para()
#
#     def forward(self, reviews, ids, ids_list):
#         """
#         reviews: [batch, review_count, review_length]
#         ids: [batch] - user o item IDs
#         ids_list: [batch, max_item_per_user] - ID degli item per user o utenti per item
#         """
#
#         # 1️⃣ Embedding parole
#         reviews = self.word_embs(reviews)  # [batch, review_count, review_length, word_dim]
#         bs, r_num, r_len, wd = reviews.size()
#         reviews = reviews.view(-1, r_len, wd)  # [batch*review_count, review_length, word_dim]
#
#         # 2️⃣ CNN sulle recensioni
#         fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)  # [batch*review_count, filters_num]
#         fea = F.adaptive_max_pool1d(fea, 1).squeeze(2)  # [batch*review_count, filters_num]
#         fea = fea.view(bs, r_num, -1)  # [batch, review_count, filters_num]
#
#         # 3️⃣ Embedding ID
#         id_emb = self.id_embedding(ids)  # [batch, id_emb_size]
#         u_i_id_emb = self.u_i_id_embedding(ids_list)  # [batch, max_item_per_user, id_emb_size]
#
#         # 4️⃣ Pooling su u_i_id_emb
#         u_i_id_emb_pooled = u_i_id_emb.mean(dim=1, keepdim=True)  # [batch, 1, id_emb_size]
#         u_i_id_emb_pooled = u_i_id_emb_pooled.expand(-1, r_num, -1)  # [batch, review_count, id_emb_size]
#
#         # 5️⃣ Linear + attenzione
#         rs_mix = F.relu(self.review_linear(fea) + self.id_linear(F.relu(u_i_id_emb_pooled)))
#         att_score = self.attention_linear(rs_mix)  # [batch, review_count, 1]
#         att_weight = F.softmax(att_score, dim=1)  # [batch, review_count, 1]
#
#         r_fea = (fea * att_weight).sum(dim=1)  # [batch, filters_num]
#         r_fea = self.fc_layer(r_fea)  # [batch, id_emb_size]
#         r_fea = self.dropout(r_fea)
#
#         # 6️⃣ ID embedding + review embedding
#         return torch.stack([id_emb, r_fea], dim=1)  # [batch, 2, id_emb_size]
#
#     def reset_para(self):
#         nn.init.xavier_normal_(self.cnn.weight)
#         nn.init.constant_(self.cnn.bias, 0.1)
#         nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)
#         nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
#         nn.init.constant_(self.review_linear.bias, 0.1)
#         nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
#         nn.init.constant_(self.attention_linear.bias, 0.1)
#         nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
#         nn.init.constant_(self.fc_layer.bias, 0.1)
#         nn.init.uniform_(self.id_embedding.weight, -0.1, 0.1)
#         nn.init.uniform_(self.u_i_id_embedding.weight, -0.1, 0.1)
