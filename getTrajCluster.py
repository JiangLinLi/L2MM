
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import torch.nn.functional as F
import numpy
import time


# def evaluate(src, model, max_length):
def evaluate(src, model, max_length, cluster_size, latten_mu_c, latten_log_sigma_sq_c):
    m0, m1 = model
    m0.eval()
    length = len(src)
    src = Variable(torch.LongTensor(src))
    src = src.view(-1, 1)
    length = Variable(torch.LongTensor([[length]]))
    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)
    z = h[-1]  #[hid_size]
    z = z.squeeze()
    print(z.shape)
    # 计算每个trajectory的label
    stack_z = torch.stack([z] * cluster_size, dim=0)  # z = [cluster_size, hid_size]
    print(stack_z.shape)
    att_logits = torch.sum(torch.pow(stack_z - latten_mu_c, 2), dim=-1)
    att_logits = att_logits.squeeze()
    att = F.softmax(att_logits)
    att_index = torch.argmin(att, dim=-1)
    return z, att_index



def clusteror(args):
    file1 = open('.\data\Freq30_cell_data_Porto_100000.txt')

    linessrc = []
    x = 0
    for line in file1:
        # print(line,x)
        if line=='E':
            print('E',line, x)
        else:
            linessrc.append(line.strip('\n').split(' '))
        x +=1

    file3 = open('./trajectory_embedding_cq.txt','w')
    file4 = open('./uc_cq.txt','w')
    file5 = open('./sig_c_cq.txt','w')
    file6 = open('./label_cq.txt','w')

    m0 = EncoderDecoder(args.input_cell_size, args.output_road_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.de_layer,
                        args.dropout, args.bidirectional, args.pretrained_embedding)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())
    #m1 = WordEmbedding(args.hidden_size, args.output_road_size)
    if os.path.isfile(args.best):
        print("=> loading checkpoint '{}'".format(args.best))
        checkpoint = torch.load(args.best, map_location='cpu')
        m0.load_state_dict(checkpoint["m0"])
        checkpoint_kmeans = torch.load('checkpoint_keans_cq.pt')
        latten_mu_c = torch.from_numpy(checkpoint_kmeans["init_mu_c"])
        latten_log_sigma_sq_c = torch.from_numpy(checkpoint_kmeans["init_sigma_c"])

        # # #
        with torch.no_grad():
            for i in range(args.cluster_size):
                #uc = str(m0.latten.mu_c[i, :].cpu().detach().numpy().tolist()).replace('[', '')
                uc = str(latten_mu_c.tolist()).replace('[','')
                uc = uc.replace(']','')
                file4.write(uc+'\n')
                sigc = str(latten_log_sigma_sq_c.tolist()).replace('[', '')
                sigc = sigc.replace(']', '')
                file5.write(sigc + '\n')

            file4.close()
            file5.close()
        i=0
        while i<1:
            try:
                print("> ", end="")
                count=0
                p = 0
                r = 0
                t=0
                for line in linessrc:
                    line1 = [int(x) for x in line]
                    print('line1',line1)
                    with torch.no_grad():
                        trg,lable = evaluate(line1, (m0, m1), args.max_length, args.cluster_size, latten_mu_c, latten_log_sigma_sq_c)
                        trg = trg.numpy().tolist()
                        print(trg)
                        print(lable)

                        trg = str(trg).replace('[','')
                        trg = str(trg).replace(']','')

                        file3.write(trg+'\n')
                        file6.write(str(lable.item())+'\n')

                    print('trg',trg)
                    print(len(trg))
                i=i+1
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))


