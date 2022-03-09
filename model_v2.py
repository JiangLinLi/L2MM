import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from SG_model import SkipGramModel


device = torch.device("cuda:0")


class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):
        super(Encoder, self).__init__()
        self.input_size = input_size
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
        #embed,_,_ = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        #embed = embed.cuda(torch.device("cuda:1"))
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output



class Latent(nn.Module):
    def __init__(self, cluster_size, hidden_size, batch_size):
        # mu_z_w = torch.Tensor(hidden_size, hidden_size)
        # nn.init.normal_(mu_z_w, std = 0.02)
        # mu_z_w.requires_grad = True
        # mu_z_w = Variable(mu_z_w).to(device)
        # mu_z_b = torch.Tensor(hidden_size)
        # nn.init.constant_(mu_z_b, 0.0)
        # mu_z_b.requires_grad = True
        super(Latent, self).__init__()
        self.cluster_size = cluster_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        #
        # self. mu_c = torch.zeros(cluster_size, hidden_size).to(device)
        # self. log_sigma_sq_c = torch.zeros(cluster_size, hidden_size).to(device)
        checkpoint_kmeans = torch.load('checkpoint_keans.pt')
        self.mu_c = torch.from_numpy(checkpoint_kmeans["init_mu_c"]).to(device)
        self.log_sigma_sq_c = torch.from_numpy(checkpoint_kmeans["init_sigma_c"]).to(device)


        self. stack_mu_c = torch.stack([self.mu_c] * batch_size, dim=0)
        self. stack_log_sigma_sq_c = torch.stack([self.log_sigma_sq_c] * batch_size, dim=0)

        self.cal_mu_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_mu_z.weight, std= 0.02)
        nn.init.constant_(self.cal_mu_z.bias, 0.0)

        self.cal_log_sigma_sq_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_log_sigma_sq_z.weight, std=0.02)
        nn.init.constant_(self.cal_log_sigma_sq_z.bias, 0.0)

    def batch_laten_loss(self, stack_log_sigma_sq_z, stack_mu_z, att, log_sigma_sq_z):
        avg_ = torch.mean(self.stack_log_sigma_sq_c
                          + torch.exp(stack_log_sigma_sq_z) / torch.exp(self.stack_log_sigma_sq_c)
                          + torch.pow(stack_mu_z - self.stack_mu_c, 2) / torch.exp(self.stack_log_sigma_sq_c), dim=-1)

        sum_ = torch.sum(att * avg_, dim=-1).squeeze()

        mean_ = torch.mean(1 + log_sigma_sq_z, dim=-1 ).squeeze()

        batch_latent_loss =0.5 * sum_-0.5*mean_

        cate_mean = torch.mean(att, dim = 0).squeeze()

        batch_cate_loss = torch.mean(cate_mean * torch.log(cate_mean)).squeeze()

        return batch_latent_loss, batch_cate_loss

    def vaeloss(self, mu_z, log_sigma_sq_z):
        sum_ = torch.sum(1+log_sigma_sq_z-torch.square(mu_z)-torch.exp(log_sigma_sq_z),dim=-1).squeeze()
        batch_latent_loss = -0.5 * sum_
        return batch_latent_loss

    def forward(self, encoder_final_state):

        encoder_final_state = encoder_final_state.squeeze()
        mu_z = self.cal_mu_z(encoder_final_state)
        log_sigma_sq_z = self.cal_log_sigma_sq_z(encoder_final_state)
        eps_z = torch.randn(size=log_sigma_sq_z.shape).to(device)
        #eps_z = torch.randn(size=log_sigma_sq_z.shape)
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z
        # vaeloss_ = self.vaeloss(mu_z,log_sigma_sq_z)
        # return z, vaeloss_

        #evaluate时注释
        #
        stack_mu_z = torch.stack([mu_z] * self.cluster_size, dim=1)
        stack_log_sigma_sq_z = torch.stack([log_sigma_sq_z] * self.cluster_size, dim=1)
        stack_z = torch.stack([z] * self.cluster_size, dim=1)

        att_logits = - torch.sum(torch.pow(stack_z - self.stack_mu_c, 2) / torch.exp(self.stack_log_sigma_sq_c), dim=-1)
        att_logits = att_logits.squeeze()
        att = F.softmax(att_logits)

        batch_latent_loss, batch_cate_loss = self.batch_laten_loss(stack_log_sigma_sq_z, stack_mu_z, att,
                                                                   log_sigma_sq_z)

        return z, batch_latent_loss, batch_cate_loss
        # return z



class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        q1 = self.L1(q).unsqueeze(2)
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
        self.hidden_size = hidden_size

    def forward(self, input, h, H, use_attention=True):
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        embed = self.embedding(input)
        #embed,_,_ = self.embedding(input)
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
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size,
                       hidden_size, num_layers, de_layer, dropout, bidirectional, cluster_size, batch_size, pretrain_embedding=None):
        super(EncoderDecoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_size = embedding_size

        if pretrain_embedding is None:
            #print('nn.Embedding')

            self.embedding1 = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0)
            self.embedding2 = nn.Embedding(output_vocab_size, embedding_size, padding_idx=0)
            self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                                   dropout, bidirectional, self.embedding1)
            self.latten = Latent(cluster_size, hidden_size, batch_size)
            self.decoder = Decoder(embedding_size, hidden_size, de_layer,
                                   dropout, self.embedding2)
            self.num_layers = num_layers

        else:
            # self.embedding1 = SkipGramModel(input_vocab_size, embedding_size)       #对cell进行embedding
            # self.embedding1 .load_state_dict(torch.load("cell_embedding.pth"))
            #self.embedding1 = pretrain_embedding
            self.embedding1 = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0)
            self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                                   dropout, bidirectional, self.embedding1)
            checkpoint = torch.load('checkpoint.pt')
            self.encoder.load_state_dict(checkpoint["encoder_m0"])
            self.latten = Latent(cluster_size, hidden_size, batch_size)
            # self.latten.load_state_dict(checkpoint["latent_m0"])
            self.embedding2 = nn.Embedding(output_vocab_size, embedding_size, padding_idx=0)
            self.decoder = Decoder(embedding_size, hidden_size, de_layer,
                                   dropout, self.embedding2)
            self.num_layers = num_layers

        # self.encoder = Encoder(embedding_size, hidden_size, num_layers,
        #                        dropout, bidirectional, self.embedding1)
        # self.latten = Latent(cluster_size, hidden_size, batch_size)
        # self.decoder = Decoder(embedding_size, hidden_size, num_layers,
        #                        dropout, self.embedding2)
        # self.num_layers = num_layers


    def encoder_hn2decoder_h0(self, h):
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        # print("encoder:{}".format(torch.cuda.memory_allocated(0)))
        z, batch_latent_loss, batch_cate_loss = self.latten(decoder_h0[-1].unsqueeze(0))
        # z, batch_latent_loss = self.latten(decoder_h0[-1].unsqueeze(0))
        # print("latten:{}".format(torch.cuda.memory_allocated(0)))
        # z = self.latten(encoder_hn[-1].unsqueeze(0))
        z = z.unsqueeze(0)

        # print(trg)
        # print(trg[:-1])
        output, decoder_hn = self.decoder(trg, z, H)
        # print("decoder:{}".format(torch.cuda.memory_allocated(0)))

        return output, batch_latent_loss, batch_cate_loss
        # return output, batch_latent_loss
        #return output

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

