import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.cfg = config
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embed)
        self.rnn = nn.GRU(self.cfg.n_embed, self.cfg.n_feature_dim, self.cfg.n_rnn_layers, batch_first=True, bidirectional=self.cfg.bidirectional)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_inds, input_masks):
        """
                Args:
                    - **input_inds**  (bsize, slen)
                    - **input_msks**  (bsize, slen)
                Returns:
                    - **feat_seq**   (bsize, slen, hsize)
                    - **last_feat**  (bsize, hsize)
                    - **hidden** [list of](num_layers * num_directions, bsize, hsize)
                """
        # Embed word ids to vectors (hacky clamp)
        lengths = torch.sum(input_masks, -1).clamp(min=1).long()
        sorted_lengths,sorted_indices = torch.sort(lengths,descending=True)
        _, unsorted_indices = torch.sort(sorted_indices)

        x = self.embed(input_inds[sorted_indices])
        packed = pack_padded_sequence(x, sorted_lengths, batch_first=True)
        self.rnn.flatten_parameters()
        features, hiddens = self.rnn(packed)
        feat_seq, _ = pad_packed_sequence(features, batch_first=True, total_length=input_masks.size(1))
        if self.cfg.bidirectional:
            N = feat_seq.size(2) // 2
            feat_seq =(feat_seq[:,:,:N] + feat_seq[:,:,N:]) / 2
        feat_seq = feat_seq[unsorted_indices]
        hiddens = hiddens[:, unsorted_indices, :]

        I = lengths.view(-1, 1, 1)
        I = I.expand(x.size(0), 1,self.cfg.n_feature_dim) - 1
        last_feat =torch.gather(feat_seq, 1, I).squeeze(1)
        return feat_seq, last_feat, hiddens
