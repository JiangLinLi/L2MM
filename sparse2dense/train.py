import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from model import EncoderDecoder
import time, os, shutil, logging
import numpy as np
import json
from torch.utils.data import DataLoader
from trajectory_dataset import Trajectory, collate_fn, convert_tensor
from util import DenseLoss
from evaluate import evaluator


def get_grids(min_lat, min_lon, max_lat, max_lon, height, width, grid_size=100):
    x_size = int(width/grid_size)
    y_size = int(height/grid_size)
    lon_vec = np.linspace(min_lon, max_lon, x_size)
    lat_vec = np.linspace(min_lat, max_lat, y_size)
    return lon_vec, lat_vec


def point2grid(lon, lat, lon_vec, lat_vec):
    lon_i = len(lon_vec[lon_vec < lon])
    lat_i = len(lat_vec[lat_vec < lat])
    if lon_i == 0 or lat_i == 0 or lon_i == len(lon_vec) or lat_i == len(lat_vec):
        return -1
    else:
        return (lat_i - 1)*(len(lon_vec)-1)+lon_i


def data_prepare(file, lat_vec, lon_vec):
    res = []
    with open(file) as f:
        for line in f:
            data = json.loads(line)
            points = data['locs']
            if len(points) > 1000:
                continue
            grids = []
            flag = 1
            index = point2grid(points[0][0], points[0][1], lon_vec, lat_vec)
            if index == -1:
                flag = 0
            else:
                grids.append(index)
            pre_lon, pre_lat = points[0][0], points[0][1]
            pre_index = index
            for i in range(1, len(points)):
                cur_lon, cur_lat = points[i][0], points[i][1]
                index = point2grid(cur_lon, cur_lat, lon_vec, lat_vec)
                if index == -1:
                    flag = 0
                else:
                    if pre_index != index:
                        grids.append(index)
                        pre_index = index
            if flag:
                res.append(grids)
    return res


def get_data_loaders(mydataset, batch_size):
    loader = DataLoader(mydataset,
                        batch_size= batch_size,
                        collate_fn=collate_fn,
                        shuffle=False)
    loader._dataset_kind = None
    return loader


def savecheckpoint(state, file_name='sparse2dense.pt'):
    torch.save(state, file_name)


def validate(valData, model, loss_f, args):
    m0, m1 = model
    m0.eval()
    m1.eval()
    loss = 0
    with torch.no_grad():
        for x, y in valData:
            if args.cuda and torch.cuda.is_available():
                x, y = (convert_tensor(x, device="cuda"), convert_tensor(y, device="cuda"))
            input, lengths = x
            input = torch.transpose(input, 0, 1)
            target, mask = torch.transpose(0, 1)
            mask = torch.transpose(mask, 0, 1)
            output = m0(input, lengths, target)
            output = m1(output)
            loss += loss_f(output, (target, mask)).item()/lengths.size(0)
    m0.train()
    m1.train()
    return loss


def train(args):
    lat_vec, lon_vec = get_grids(args.min_lat, args.min_lon, args.max_lat, args.max_lon, args.height, args.width)
    data_file = './data/trajectory.json'
    args.input_cell_size = len(lat_vec)*len(lon_vec)+3
    args.output_cell_size = len(lat_vec)*len(lon_vec)+3
    all_data = data_prepare(data_file, lat_vec=lat_vec, lon_vec=lon_vec)
    all_dataset = Trajectory(all_data)
    train_size = int(0.8 * len(all_dataset))
    val_size = int((len(all_dataset)-train_size) / 2)
    test_size = len(all_dataset)-train_size-val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_dataset,
                                                                             [train_size, val_size, test_size])
    trainData = get_data_loaders(train_dataset, args.batch)
    valData = get_data_loaders(val_dataset, args.batch)

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
    loss_f = DenseLoss()
    if args.cuda and torch.cuda.is_available():
        m0.to("cuda")
        m1.to("cuda")
        loss_f.to("cuda")

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

    early_stop = False
    early_count = 0
    for epoch in range(args.epochs):
        if early_stop:
            break
        epoch_loss = 0
        for x, y in trainData:
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            if args.cuda and torch.cuda.is_available():
                x, y = (convert_tensor(x, device="cuda"), convert_tensor(y, device="cuda"))
                input, lengths = x
                input = torch.transpose(input, 0, 1)
                target, mask = torch.transpose(0, 1)
                mask = torch.transpose(mask, 0, 1)
                output = m0(input, lengths, target)
                output = m1(output)
                loss = loss_f(output, (target, mask))
                loss.backward()
                clip_grad_norm(m0.parameters(), 5)
                clip_grad_norm(m1.parameters(), 5)
                m0_optimizer.step()
                m1_optimizer.step()

                epoch_loss += loss.item()/lengths.size(0)

        val_loss = validate(valData, (m0, m1), loss_f, args)
        print("epoch: {}, train_loss: {}, val_loss: {}".format(epoch, epoch_loss, val_loss))

        if val_loss < best_prec_loss:
            best_prec_loss = val_loss
            early_count = 0
            savecheckpoint({
                "epoch": epoch,
                "val_loss": best_prec_loss,
                "encoder_m0": m0.encoder.state_dict(),
                "m0": m0.state_dict(),
                "m1": m1.state_dict(),
                "m0_optimizer": m0_optimizer.state_dict(),
                "m1_optimizer": m1_optimizer.state_dict()
            })
        else:
            early_count += 1
            if early_count >args.patience:
                early_stop = True
                print("Early stopped")
            else:
                print("EarlyStopping: {} / {}".format(early_count, args.patience))

    print("begin evaluate")
    evaluator(test_dataset, args)
