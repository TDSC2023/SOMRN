from torch import optim
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from utils.utils import (ParameterContainer,
                         Recorder,
                         Dataset)
from torch.utils.data import DataLoader
from utils.models import LSTMClassifier

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='set your mode: cv|train')
parser.add_argument('-G', '--GPU', default='3', help='GPU to be used')
parser.add_argument('-f', '--fold', default=5, type=int, help='the number of folds for CV or training')
parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of DataLoader')
args = parser.parse_args()

GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

pc = ParameterContainer(
    epochs=40,
    in_features=246,
    hidden_neurons=128,
    batch_size=32, 
    category=8,
    lr=1e-3, 
    seq_len=300,
    num_layers=1,
    vector_type='one-hot',
    # vector_type='reduced',
    index2vec=None,
    # index2vec='PPOfirst', 
)

base_log_folder = './2_layer_lstm_log/'
if not os.path.exists(base_log_folder):
    os.mkdir(base_log_folder)

base_name = '2_layer_lstm_{}_{}_{}_{}'.format(pc.in_features, pc.hidden_neurons, pc.category, pc.index2vec)

def get_conf_matrix(classifier, dataset):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    classifier.eval()
    conf_matrix = {} 
    total = len(dataset)
    correct = 0
    with torch.no_grad():
        for index, (data, label) in enumerate(dataloader):
            predict = torch.softmax(classifier(data.to(device)), dim=1)
            real_label = label.item()
            predict_label = torch.argmax(predict.cpu(), dim=1).item()
            if real_label not in conf_matrix:
                conf_matrix[real_label] = [0 for _ in range(pc.category)]
            conf_matrix[real_label][predict_label] += 1
            if real_label == predict_label:
                correct += 1
    conf_matrix['acc'] = correct / total
    return conf_matrix


def training_alg(dataset, Loss, fold, mode):
    dataset.set_mode('train_set')
    T = 0
    if mode == 'cv':
        T += (len(dataset) / pc.batch_size + 1) * (fold - 1) * pc.epochs
    else:  
        T += (len(dataset) / pc.batch_size + 1) * fold * pc.epochs
    path = base_log_folder + base_name + '_{}_details.json'.format(mode)
    pc.parameters['fold'] = fold
    recorder = Recorder(path,
                        pc.parameters,
                        arg_dicts=[{'name': 'fold', 'max_value': fold},
                                   {'name': 'epoch', 'max_value': pc.epochs}],
                        T=T)
    stat_dict = None if mode == 'cv' else dict()
    acc = []
    for k in range(fold):
        if mode == 'cv':
            dataset.set_mode('cross_validation', validate=False, k=k)
        else:
            dataset.set_mode('train_set')
        dataloader = DataLoader(dataset, batch_size=pc.batch_size, num_workers=args.num_workers, shuffle=True)
        lstm = LSTMClassifier(
            in_features=pc.in_features,
            hidden_neurons=pc.hidden_neurons,
            num_layers=pc.num_layers,
            category=pc.category,
        ).to(device)
        optimizer = optim.Adam(lstm.parameters(), lr=pc.lr)
        lstm.train()
        for epoch in range(pc.epochs):
            for index, (data, label) in enumerate(dataloader):
                data, label = data.to(device), label.flatten().to(device)
                predict = lstm(data)
                loss = Loss(predict, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                recorder.write(k, epoch, loss=loss.cpu().item())
            recorder.reduce(k, epoch, func=np.mean)
        if mode == 'cv':
            dataset.set_mode('cross_validation', validate=True, k=k)
        else:
            dataset.set_mode('test_set')
        conf_matrix = get_conf_matrix(lstm, dataset)
        acc.append(conf_matrix['acc'])
        recorder.record['fold_%d' % k]['conf_matrix'] = conf_matrix
        if stat_dict is not None:
            stat_dict['fold_%d' % k] = lstm.cpu().state_dict()
    recorder.record['avg_acc'] = np.mean(acc)
    recorder.record['std'] = np.std(acc)
    recorder.save()
    if stat_dict is not None:
        torch.save(
            obj=stat_dict,
            f=base_log_folder + base_name + '.pt'
        )


if __name__ == "__main__":
    dataset = Dataset(dataset_name='api-2019',
                      csv_file='dataset/api-2019/api-2019.csv',
                      seq_len=10,
                      fmt=['category', 'name', 'length', 'seq'])
    dataset.initialize_word2index('./dataset/word2index.json')
    index2vec = None if pc.index2vec is None else './dataset/' + pc.index2vec + '.pickle'
    dataset.set_vector_type(pc.vector_type, index2vec=index2vec)
    # class_weight = torch.FloatTensor(dataset.compute_class_weight()).cuda()
    # CELoss = nn.CrossEntropyLoss(weight=class_weight)
    CELoss = nn.CrossEntropyLoss()
    if args.mode in ('cv', 'train'):
        training_alg(dataset, CELoss, fold=args.fold, mode=args.mode)
    else:
        raise Exception('undefined mode')
