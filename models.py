import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from cell_embedding import Cell_Embedding
from SG_model import SkipGramModel

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
        #print(input)
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output


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
        return c
        # c = torch.cat([c, q], 1)
        # return self.tanh(self.L2(c))


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
                o_ = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            o = torch.cat((o,o_), dim=1)
            output.append(o)
        output = torch.stack(output)
        return output, h


class EncoderDecoder(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size,
                       hidden_size, num_layers, de_layer, dropout, bidirectional,  pretrain_embedding=None):
        super(EncoderDecoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_size = embedding_size
        self.dest_embedding = nn.Embedding(input_vocab_size, hidden_size)

        #self.embedding1 = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0)
        #self.embedding1 = Cell_Embedding(input_vocab_size,embedding_size,input_vocab_size)
        if pretrain_embedding is None:
            #print('nn.Embedding')

            self.embedding1 = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0)
            self.embedding2 = nn.Embedding(output_vocab_size, embedding_size, padding_idx=0)
            self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                                   dropout, bidirectional, self.embedding1)
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
            self.embedding2 = nn.Embedding(output_vocab_size, embedding_size, padding_idx=0)
            self.decoder = Decoder(embedding_size, hidden_size, de_layer,
                                   dropout, self.embedding2)
            self.num_layers = num_layers




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
        # dest_embed = self.dest_embedding(dest)
        # stack_dest = torch.stack([dest_embed]*encoder_hn.shape[0], dim=0)

        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        decoder_h0 = decoder_h0[-1].unsqueeze(0)

        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output

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
        # input = [bathch, hid_dim]
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i   # 输出最后一层
        hn = torch.stack(hn)    # 三层的结果
        return output, hn

