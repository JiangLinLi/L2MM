import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from models import EncoderDecoder
from data_utils import DataLoader
import time, os, shutil, logging, h5py
from inputData import InputData


torch.backends.cudnn.enabled = False
device = torch.device("cuda:1")

def NLLcriterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[0] = 0
    criterion = nn.NLLLoss(weight, ignore_index=0,size_average=False)
    print(criterion)
    return criterion


def KLDIVloss(output, target, criterion, V, D):
    """
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    ## (batch, k) index in vocab_size dimension
    ## k-nearest neighbors for target
    indices = torch.index_select(V, 0, target)          #找出target行中相应的邻居
    ## (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)    #选择邻居对应的向量
    outputk = outputk.log_softmax(dim=1)

    ## (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)      #找出对应的权重
    return criterion(outputk, targetk)

def TrajModelLoss(output,target,Adj_mat,Adj_mask,generator,batch_mask, D):
    #output = [t,batch,hid_dim]
    #target = [batch,t+1]
    batch = output.size(1)
    batch_mask_flat = batch_mask[:,1:]
    input_flat = target[:, :-1]
    target_flat = target[:, 1:]
    output_flat_ = output.view(-1, output.size(2))   #[batch*t, hid_dim], float
    target_flat = target_flat.reshape(-1)    #[batch*t]
    input_flat_ = input_flat.reshape(-1)   #[batch*t]
    target_flat_ = target_flat.unsqueeze(dim=1) # [batch*t,1]
    sub_adj_mat_ = torch.index_select(Adj_mat,0,input_flat_)     #按行索引 [batch*t, max_adj_num]
    sub_adj_mask_ = torch.index_select(Adj_mask,0,input_flat_)
    #targetk = torch.index_select(D,0,target_flat)       # [batch*t,mac_adj_num]
    # first column is target_
    target_and_sub_adj_mat_ = torch.cat((target_flat_, sub_adj_mat_),1)
    target_and_sub_adj_mat_ = target_and_sub_adj_mat_.reshape(-1)
    outputs_3d_ = output_flat_.unsqueeze(dim=1) # [batch*t, hid_dim] -> [batch*t, 1, hid_dim]
    w_t = generator.weight
    b_t = generator.bias
    # w_t = generator.state_dict()['0.weight']        # [state_size,hid_dim]
    # b_t = generator.state_dict()['0.bias']      # [state_size]
    sub_w_flat_ = torch.index_select(w_t,0,target_and_sub_adj_mat_)     # [batch*t*max_adj_num+1,hid_dim]
    sub_b_ = torch.index_select(b_t,0,target_and_sub_adj_mat_)   # [batch*t*max_adj_num+1]
    sub_b_flat_ = sub_b_.unsqueeze(dim=1)    # [batch*t*max_adj_num+1,1]
    outputs_tiled_ = outputs_3d_.repeat(1,Adj_mat.shape[1]+1,1) # [batch*t, max+adj_num+1, hid_dim]
    outputs_tiled_ = outputs_tiled_.reshape(-1,int(outputs_tiled_.shape[2]))     # [batch*t*max_adj_num+1, hid_dim]
    target_logit_and_sub_logits_ = torch.reshape(torch.sum(torch.add(torch.mul(sub_w_flat_,outputs_tiled_),sub_b_flat_),dim =1),[-1,Adj_mat.shape[1]+1])    # [batch*t, max_adj_num+1]
    # scales_ =torch.max(target_logit_and_sub_logits_, dim=1)[0]  # [batch*t]
    # scaled_target_logit_and_sub_logits_ = torch.sub(target_logit_and_sub_logits_.t(),scales_).t()
    scaled_target_logit_and_sub_logits_ = target_logit_and_sub_logits_
    scaled_sub_logits_ = scaled_target_logit_and_sub_logits_[:, 1:]  # [batch*t, max_adj_num]
    exp_scaled_sub_logits_ = torch.exp(scaled_sub_logits_)  # [batch*t, max_adj_num]        #求指数
    sum_exp_scaled_sun_logits = torch.sum(exp_scaled_sub_logits_,dim=1)         #求和
    deno_ = torch.div(exp_scaled_sub_logits_.t(), sum_exp_scaled_sun_logits).t()         #归一化
    #deno_ = torch.mul(targetk,deno_)
    test = torch.sum(deno_,dim=1)
    deno_ = torch.sum(torch.mul(deno_, sub_adj_mask_), 1)  # [batch*t]
    log_deno_ = torch.log(deno_)  # [batch*t] 预测的input的邻居的分布
    nume_ = scaled_target_logit_and_sub_logits_[:, 0:1]
    exp_nume_ = torch.exp(nume_)
    nume_ = torch.div(exp_nume_.t(),sum_exp_scaled_sun_logits).t()
    log_nume_ = torch.reshape(nume_,[-1]).log()

    loss_ = log_nume_
    masked_loss_p_ = torch.mul(loss_, batch_mask_flat.reshape(-1))  # [batch*t]
    loss_p_ = -1*torch.sum(masked_loss_p_) / batch
    return loss_p_

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

def savecheckpoint(state, is_best, filename="mmcheckpoint.pt"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_model.pt')

def validate(valData, model, lossF, args, inputdata,Adj_mat,Adj_mask,D):
    m0, m1 = model
    m0.eval()
    m1.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_loss = 0
    for iteration in range(num_iteration):
        input, lengths, target, batch_mask = valData.getbatch()
        dest = []
        len_ = lengths.squeeze()
        for i in range(input.shape[1]):
            dest.append(input[len_[i] - 1, i])
        dest = torch.LongTensor(dest)
        with torch.no_grad():
            input = Variable(input)
            lengths = Variable(lengths)
            target = Variable(target)
            batch_mask = Variable(batch_mask)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target, batch_mask, dest = input.to(device), lengths.to(device), target.to(device), batch_mask.to(device), dest.to(device)


            output = m0(input, lengths, target, dest)
            # target = target.t()
            # batch_mask = batch_mask.t()
            # loss = TrajModelLoss(output,target,Adj_mat,Adj_mask,m1,batch_mask,D)
            loss = batchloss(output, target, m1, lossF, 1,inputdata)
            total_loss += loss * output.size(1)
    m0.train()
    m1.train()
    return total_loss.item() / valData.size

def train(args):
    logging.basicConfig(filename="training.log", level=logging.INFO)
    trainsrc = '.\data\Freq90_cell_data_porto_train.txt'
    traintrg = '.\data\Freq15_cell_data_porto_train.txt'

    trainData = DataLoader(trainsrc, traintrg, args.batch, args.bucketsize)
    print("Read training data")
    trainData.load(args.max_num_line)

    valsrc = '.\data\Freq90_cell_data_porto_val.txt'
    valtrg = '.\data\Freq15_cell_data_porto_val.txt'

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
        criterion = nn.KLDivLoss(reduction='sum')
        lossF = lambda o, t: KLDIVloss(o, t, criterion, Adj_mat,Adj_mask)

    m0 = EncoderDecoder(args.input_cell_size,
                        args.output_road_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.de_layer,
                        args.dropout,
                        args.bidirectional,
                        args.pretrained_embedding)

    m1 = nn.Sequential(nn.Linear(2*args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())
    print(args.cuda)
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
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
        print("no checkpoint".format(args.checkpoint))
        logging.info("Start training".format(time.ctime()))
        best_prec_loss = float('inf')
    print(sum(trainData.allocation))
    num_iteration = args.epochs * sum(trainData.allocation) // args.batch
    print("Start at {} "
          "and end at {}".format(args.start_iteration, num_iteration-1))
    for iteration in range(args.start_iteration, num_iteration):
        try:
            input, lengths, target, batch_mask= trainData.getbatch()
            dest = []
            len_ = lengths.squeeze()
            for i in range(input.shape[1]):
                dest.append(input[len_[i]-1,i])
            dest = torch.LongTensor(dest)
            #dest = dest.unsqueeze(dim=0)
            input, lengths, target, batch_mask, dest = Variable(input), Variable(lengths), Variable(target),Variable(batch_mask), Variable(dest)
            #print(input.shape, lengths.shape, target.shape)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target, batch_mask, dest = input.to(device), lengths.to(device), target.to(device), batch_mask.to(device), dest.to(device)

            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()

            try:
                output = m0(input, lengths, target)
                #output, batch_latent_loss, batch_cate_loss = m0(input, lengths, target)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            loss = batchloss(output, target, m1, lossF, args.g_batch)
            loss.backward()
            clip_grad_norm(m0.parameters(), 5)
            clip_grad_norm(m1.parameters(), 5)
            m0_optimizer.step()
            m1_optimizer.step()
            avg_loss = loss.item() / target.size(0)
            if iteration % args.print == 0:
                print("Iteration: {}\tLoss: {}".format(iteration, avg_loss))

            if iteration % args.save == 0 and iteration > 0:
               is_best = True
               savecheckpoint({
                   "iteration": iteration,
                   "best_prec_loss": best_prec_loss,
                   "m0": m0.state_dict(),
                   "m1": m1.state_dict(),
                   "m0_optimizer": m0_optimizer.state_dict(),
                   "m1_optimizer": m1_optimizer.state_dict()
               }, is_best)
        except KeyboardInterrupt:
            break