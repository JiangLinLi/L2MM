import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from model import EncoderDecoder
from data_util import DataLoader
import time, os, shutil, logging
from init_latent import pretrain_bucket
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


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

def savecheckpoint(state, filename="trajectory2path.pt"):
    torch.save(state, filename)

def validate(valData, model, lossF, args):
    m0, m1 = model
    m0.eval()
    m1.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_loss = 0
    for iteration in range(num_iteration):
        input, lengths, target, batch_mask = valData.getbatch()
        with torch.no_grad():
            input = Variable(input)
            lengths = Variable(lengths)
            target = Variable(target)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target = input.to(device), lengths.to(device), target.to(device)
            output, batch_gaussian_loss, batch_cate_loss = m0(input, lengths, target)
            loss = batchloss(output, target, m1, lossF, 2)
            if args.cluater_size == 1:
                loss = loss + batch_gaussian_loss
            else:
                loss = loss + 1.0/args.hidden_size * batch_gaussian_loss + 0.1 * batch_cate_loss
            total_loss += loss * output.size(1)
    m0.train()
    m1.train()
    return total_loss.item() / valData.size

def evaluate(src, model, max_length):
    m0, m1 = model
    length = len(src)
    src = Variable(torch.LongTensor(src))
    length = Variable(torch.LongTensor([[length]]))
    en_hn, H = m0.encoder(src, length)
    de_h0 = m0.encoder_hn2decoder_h0(en_hn)
    z = m0.latent(de_h0[-1].unsqueeze(0), 'test')
    h = z.unsqueeze(0).unsqueeze(0)
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


def evaluator(args, filename="trajectory2path.pt"):
    test_src = ".\data\Freq90_cell_data_porto_test.txt"
    test_trg = ".\data\Freq15_path_data_porto_test.txt"
    src = []
    for line in test_src:
        line = line.strip('\n').split(' ')
        line = [int(x) for x in line]
        src.append(line)
    trg = []
    for line in test_trg:
        line = line.strip('\n').split(' ')
        line = [int(x) for x in line]
        trg.append(line)


    m0 = EncoderDecoder(
        args.cluster_size,
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
        test_len = len(src)
        count = 0
        p = 0
        r = 0
        t = 0
        for i in range(test_len):
            x, y = src[i], trg[i]
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



def train_bucket(args):
    pretrain_bucket(args)

    trainsrc = '.\data\Freq90_cell_data_porto_train.txt'
    traintrg = '.\data\Freq15_path_data_porto_train.txt'

    trainData = DataLoader(trainsrc, traintrg, args.batch, args.bucketsize)
    print("Read training data")
    trainData.load(args.max_num_line)

    valsrc = '.\data\Freq90_cell_data_porto_val.txt'
    valtrg = '.\data\Freq15_path_data_porto_val.txt'

    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, args.batch, args.bucketsize, True)
        print("Read validation data")
        valData.load()
        print("Load validation data")
    else:
        print("No validation data")

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

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        best_prec_loss = checkpoint["val_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
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

        output, batch_gaussian_loss, batch_cate_loss = m0(input, lengths, target)
        loss = batchloss(output, target, m1, lossF, 2)
        if args.cluater_size == 1:
            loss = loss + batch_gaussian_loss
        else:
            loss = loss + 1.0/args.hidden_size * batch_gaussian_loss + 0.1 * batch_cate_loss
        loss.backward()
        clip_grad_norm(m0.parameters(), 5)
        clip_grad_norm(m1.parameters(), 5)
        m0_optimizer.step()
        m1_optimizer.step()
        avg_loss = loss.item() / target.size(0)
        if iteration % args.print == 0:
            print("Iteration: {}\tLoss: {}".format(iteration, avg_loss))

        if iteration % args.save == 0 and iteration > 0:
            val_loss = validate(valData, (m0, m1), lossF, args)
            print("Iteration: {}, train_loss: {}, val_loss: {}".format(iteration, avg_loss, val_loss))
            
            if val_loss < best_prec_loss:
                best_prec_loss = val_loss
                early_count = 0
                savecheckpoint({
                    "iteration": iteration,
                    "val_loss": best_prec_loss,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                })
            else:
                early_count += 1
                if early_count<args.patience:
                    early_stop = True
                    print('Early stopped')
                else:
                    print("Early stopping {} / {}".format(early_count, args.patience))
