import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from model import EncoderDecoder
from data_util import DataLoader
import time, os, shutil, logging
import numpy as np
from sklearn.cluster import KMeans
device = torch.device("cuda:0")


def batchloss(output, target, generator, lossF, g_batch):
    batch = output.size(1)
    loss = 0
    target = target[1:]
    for o, t in zip(output.split(g_batch),
                    target.split(g_batch)):
        o = o.view(-1, o.size(2))
        o = generator(o)
        t = t.view(-1)
        loss += lossF(o, t)
    return loss.div(batch/g_batch)


def savecheckpoint(state, filename="pretrain_latent.pt"):
    torch.save(state, filename)

def save_mu_c(args, m0, filename="pretrain_latent.pt"):
    test_src = ".\data\Freq90_cell_data_porto_train.txt"
    test_trg = ".\data\Freq15_path_data_porto_train.txt"
    trainData = DataLoader(test_src, test_trg, args.batch, args.bucketsize)
    print("Read training data")
    trainData.load(10000)
    num_iteration = sum(trainData.allocation) // args.batch

    with torch.no_grad():
        x_embedded = []
        for iteration in range(num_iteration):
            input, lengths, target = trainData.getbatch()
            input, lengths, target = Variable(input), Variable(lengths), Variable(target)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target = input.to(device), lengths.to(device), target.to(device)

            z = m0(input, lengths, target, 'test')

            z = z.unsqueeze(0)
            x_embedded.append(z)

        x_embedded = torch.cat(x_embedded, dim=0)
        x_embedded = torch.reshape(x_embedded,(-1,args.hidden_size))
        x_embedded = x_embedded.cpu().numpy()
        kmeans = KMeans(n_clusters=args.cluster_size)
        kmeans.fit(x_embedded)
        init_mu_c = kmeans.cluster_centers_
        savecheckpoint({
            "init_mu_c": init_mu_c
        }, filename="init_latent.pt")


def pretrain_bucket(args):
    trainsrc = '.\data\Freq90_cell_data_porto_train.txt'
    traintrg = '.\data\Freq15_path_data_porto_train.txt'

    trainData = DataLoader(trainsrc, traintrg, args.batch, args.bucketsize)
    print("Read training data")
    trainData.load(10000)

    if args.criterion_name == "CE":
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        lossF = lambda o, t: criterion(o, t)
    else:
        criterion = nn.NLLLoss(ignore_index=0).to(device)
        lossF = lambda o, t: criterion(o, t)

    m0 = EncoderDecoder(args.cluster_size,
                        args.input_cell_size,
                        args.output_road_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.de_layer,
                        args.dropout,
                        args.bidirectional)

    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())

    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.to(device)
        m1.to(device)
        criterion.to(device)
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)
    best_prec_loss = float('inf')
    start_iteration = 0
    num_iteration = args.epochs * sum(trainData.allocation) // args.batch
    print("Start at {} "
          "and end at {}".format(start_iteration, num_iteration-1))

    early_stop = False
    early_count = 0

    for iteration in range(start_iteration, num_iteration):
        if early_stop:
            break
        input, lengths, target = trainData.getbatch()

        input, lengths, target = Variable(input), Variable(lengths), Variable(target)

        if args.cuda and torch.cuda.is_available():
            input, lengths, target = input.to(device), lengths.to(device), target.to(device)

        m0_optimizer.zero_grad()
        m1_optimizer.zero_grad()

        output = m0(input, lengths, target, kind="pretrain")
        loss = batchloss(output, target, m1, lossF, 2)
        loss.backward()
        clip_grad_norm(m0.parameters(), 5)
        clip_grad_norm(m1.parameters(), 5)
        m0_optimizer.step()
        m1_optimizer.step()
        avg_loss = loss.item() / target.size(0)
        if iteration % args.print == 0:
            print("Iteration: {}\tLoss: {}".format(iteration, avg_loss))

        if iteration % args.save == 0 and iteration > 0:
            if avg_loss < best_prec_loss:
                best_prec_loss = avg_loss
                early_count = 0
                savecheckpoint({
                    "iteration": iteration,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                })
            else:
                early_count += 1
                if early_count < args.patience:
                    early_stop = True
                    print('Early stopped')
                else:
                    print("Early stopping {} / {}".format(early_count, args.patience))
    print("init_latent")
    checkpoint = torch.load("pretrain_latent.pt")
    m0.load_state_dict(checkpoint["m0"])
    save_mu_c(args, m0)
