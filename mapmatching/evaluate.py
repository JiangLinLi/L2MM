import torch
import torch.nn as nn
from torch.autograd import Variable
from model import EncoderDecoder
import os
import time


def evaluate(src, model, max_length):
    m0, m1 = model
    length = len(src)
    src = Variable(torch.LongTensor(src))
    length = Variable(torch.LongTensor([[length]]))
    en_hn, H = m0.encoder(src, length)
    de_h0 = m0.encoder_hn2decoder_h0(en_hn)
    h = de_h0[-1].unsqueeze(0)
    input_ = Variable(torch.LongTensor([[1]]))
    tg = []
    for _ in range(max_length):
        o, h = m0.decoder(input_, h, H)
        o = o.view(-1, o.size(2))
        o = m1(o)
        _, id = o.data.topk(1)
        id = id[0][0]
        if id == 2:
            break
        tg.append(id.item())
        input_ = Variable(torch.LongTensor([[id]]))
    return tg


def evaluator(test_dataset, args, filename="trajectory2path.pt"):
    m0 = EncoderDecoder(
        args.input_cell_size,
        args.output_cell_size,
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        args.de_layer,
        args.dropout,
        args.bidirectional
    )

    m1 = nn.Sequential(
        nn.Linear(args.hidden_size, args.output_cell_size),
        nn.LogSoftmax()
    )

    if os.path.isfile(filename):
        print("loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        test_len = test_dataset.__len__()
        count = 0
        p = 0
        r = 0
        t = 0
        for i in range(test_len):
            x, y = test_dataset.__getitem__(i)
            y_pred = evaluate(x, (m0, m1), args.max_length)
            start = time.time()
            end = time.time()
            t = t + end-start
            if len(y) >0:
                print('target', y)
                print('model_prediction', y_pred)

                intersect = set(y_pred) & set(y)
                p = p + len(intersect) / max(len(y_pred), len(y))
                r = r + len(intersect) / len(y_pred)

                count = count + 1

        print("t {}".format(t / count))
        print("p {}".format(p / count))
        print("r {}".format(r / count))
    else:
        print("no checkpoint found at {}".format(filename))