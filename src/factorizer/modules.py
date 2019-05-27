import torch


class MF(torch.nn.Module):
    """Matrix Factorization"""
    def __init__(self, opt):
        super(MF, self).__init__()
        self.num_users = opt['num_users']
        self.num_items = opt['num_items']
        self.latent_dim = opt['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """Compute Score for ranking"""
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        score = torch.sum(element_product, dim=1)
        return score

    def forward_triple(self, u, i, j):
        """Compute BPR Score

        :param u: user index
        :param i: positive item index
        :param j: negative item index
        :return: p(s(u,i) > s(u,j))
        """
        u_emb = self.embedding_user(u)
        i_emb = self.embedding_item(i)
        j_emb = self.embedding_item(j)
        ui_score = torch.mul(u_emb, i_emb).sum(dim=1)
        uj_score = torch.mul(u_emb, j_emb).sum(dim=1)
        logit = ui_score - uj_score
        return logit
        # prob = self.logistic(logit)
        # return prob

    def l2_penalty(self, l2_lambda, u, i, j):
        """Compute dimension-wise l2-penalty
                or dimension+user-wise l2-penalty
                or dimension+item-wise l2-penalty

        Args:
            l2_lambda: lambda, list of pytorch tensor
        """
        w_u_sq = self.embedding_user(u).pow(2)
        w_i_sq = self.embedding_item(i).pow(2)
        w_j_sq = self.embedding_item(j).pow(2)
        # w_u_sq = self.embedding_user.weight.pow(2)
        # w_i_sq = self.embedding_item.weight.pow(2)
        if isinstance(l2_lambda, list):
            # user, item with different lambdas
            user_penalty, item_penalty = l2_lambda[0], l2_lambda[1]
        else:
            # user, item with the same lambda
            user_penalty, item_penalty = l2_lambda, l2_lambda
        if user_penalty.dim() == 2: # (num_user, latent_dim)
            # user-wise lambda
            assert user_penalty.size()[0] == self.num_users
            u_penalty = user_penalty[u]
        else:
            # global user lambda
            u_penalty = user_penalty
        if item_penalty.dim() == 2: 
            assert item_penalty.size()[0] == self.num_items
            i_penalty = item_penalty[i]
            j_penalty = item_penalty[j]
        else:
            i_penalty = item_penalty
            j_penalty = item_penalty
        
        up = w_u_sq * u_penalty
        ip = w_i_sq * i_penalty
        jp = w_j_sq * j_penalty
        l2_reg = up.sum() + ip.sum() + jp.sum()

        # up = w_u_sq * user_penalty
        # ip = w_i_sq * item_penalty
        # l2_reg = up.sum() + ip.sum()
        l2_reg = l2_reg.sum()
        return l2_reg