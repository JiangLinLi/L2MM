import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import os
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output


class LatentDistribution(nn.Module):
    def __init__(self, cluster_size, hidden_size, path="init_latent.pt"):
        super(LatentDistribution, self).__init__()
        self.cluter_size = cluster_size
        self.hidden_size = hidden_size
        if os.path.isfile(path):
            mu_c = torch.load(path)["init_mu_c"]
            mu_c = torch.from_numpy(mu_c)
            if torch.cuda.is_available():
                mu_c = mu_c.to("cuda")
            self.mu_c = nn.Parameter(mu_c, requires_grad=True)

        else:
            mu_c = torch.rand(cluster_size, hidden_size)
            self.mu_c = nn.Parameter(mu_c, requires_grad=True)

        log_sigma_sq_c = torch.zeros(cluster_size, hidden_size)
        if torch.cuda.is_available():
            log_sigma_sq_c = log_sigma_sq_c.to("cuda")
        self.log_sigma_sq_c = nn.Parameter(log_sigma_sq_c, requires_grad=True)

        self.cal_mu_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_mu_z.weight, std=0.02)
        nn.init.constant_(self.cal_mu_z.bias, 0.0)

        self.cal_log_sigma_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_log_sigma_z.weight, std=0.02)
        nn.init.constant_(self.cal_log_sigma_z.bias, 0.0)

    def batch_laten_loss(self, stack_log_sigma_sq_c, stack_mu_c, stack_log_sigma_sq_z, stack_mu_z, att, log_sigma_sq_z):
        avg_ = torch.mean(stack_log_sigma_sq_c
                          + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
                          + torch.pow(stack_mu_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c), dim=-1)

        sum_ = torch.sum(att * avg_, dim=-1).squeeze()

        mean_ = torch.mean(1 + log_sigma_sq_z, dim=-1).squeeze()

        batch_latent_loss = 0.5 * sum_ - 0.5 * mean_
        batch_latent_loss = torch.mean(batch_latent_loss).squeeze()

        cate_mean = torch.mean(att, dim=0).squeeze()
        batch_cate_loss = torch.mean(cate_mean * torch.log(cate_mean)).squeeze()
        batch_cate_loss = torch.mean(batch_cate_loss).squeeze()

        return batch_latent_loss, batch_cate_loss

    def forward(self, h, kind="train"):
        h = h.squeeze()
        mu_z = self.cal_mu_z(h)
        if kind == "test":
            return mu_z
        log_sigma_sq_z = self.cal_log_sigma_z(h)
        eps_z = torch.rand(size=log_sigma_sq_z.shape)
        if kind == "pretrain" or kind=="train":
            if torch.cuda.is_available():
                eps_z = eps_z.to("cuda")

        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z

        if kind == "pretrain":
            return z
        else:
            stack_mu_c = torch.stack([self.mu_c] * z.shape[0], dim=0)
            stack_log_sigma_sq_c = torch.stack([self.log_sigma_sq_c] * z.shape[0], dim=0)
            stack_mu_z = torch.stack([mu_z] * self.cluter_size, dim=1)
            stack_log_sigma_sq_z = torch.stack([log_sigma_sq_z] * self.cluter_size, dim=1)
            stack_z = torch.stack([z] * self.cluter_size, dim=1)
            att_logits = - torch.sum(torch.pow(stack_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c), dim=-1)
            att_logits = att_logits.squeeze()
            att = F.softmax(att_logits)+ 1e-10

            batch_latent_loss, batch_cate_loss = self.batch_laten_loss(stack_log_sigma_sq_c, stack_mu_c, stack_log_sigma_sq_z, stack_mu_z, att,
                                                                       log_sigma_sq_z)
            return z, batch_latent_loss, batch_cate_loss


class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        q1 = q.unsqueeze(2)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        a = a.unsqueeze(1)
        c = torch.bmm(a, H).squeeze(1)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        embed = self.embedding(input)
        output = []
        for e in embed.split(1):
            e = e.squeeze(0)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h


class EncoderDecoder(nn.Module):
    def __init__(self, cluster_size, input_vocab_size, output_vocab_size, embedding_size,
                       hidden_size, num_layers, de_layer, dropout, bidirectional, is_pretrain=True):
        super(EncoderDecoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_size = embedding_size
        self.embedding1 = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0)
        self.embedding2 = nn.Embedding(output_vocab_size, embedding_size, padding_idx=0)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding1)
        if is_pretrain:
            filename = "sparse2dense.pt"
            if os.path.isfile(filename):
                checkpoint = torch.load(filename)
                self.encoder.load_state_dict(checkpoint["encoder_m0"])
        self.decoder = Decoder(embedding_size, hidden_size, de_layer,
                               dropout, self.embedding2)
        self.latent = LatentDistribution(cluster_size, hidden_size)
        self.num_layers = num_layers

    def encoder_hn2decoder_h0(self, h):
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg, kind="train"):
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        if kind == "train":
            z, batch_latent_loss, batch_cate_loss = self.latent(decoder_h0[-1].unsqueeze(0))
            z = z.unsqueeze(0)
            output, de_hn = self.decoder(trg[:-1], z, H)
            return output, batch_latent_loss, batch_cate_loss
        elif kind == "pretrain":
            z = self.latent(decoder_h0[-1].unsqueeze(0), kind)
            z = z.unsqueeze(0)
            output, de_hn = self.decoder(trg[:-1], z, H)
            return output
        elif kind == 'test':
            z = self.latent(decoder_h0[-1].unsqueeze(0), kind)
            return z


class StackingGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn

