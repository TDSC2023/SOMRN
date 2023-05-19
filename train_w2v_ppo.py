import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn.parallel as par
import numpy as np
import os
import pickle
import argparse
import PPO.agent
from config import *
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils.cbow import Encoder, Decoder, PolicyNetwork, RNNPolicyNetwork, LSTMPolicyNetwork, policyNet, LSTM_Encoder
from utils.utils import (ParameterContainer,
                         W2VWrapper,
                         Reward,
                         Recorder,
                         Dataset)


parser = argparse.ArgumentParser()
parser.add_argument('-beta', help='beta * K', default=0.05)
args = parser.parse_args()
# torch.set_default_tensor_type(torch.FloatTensor)
# save_train_loss_change = False
torch.autograd.set_detect_anomaly(True)
# CUDA_LAUNCH_BLOCKING = 1
device = DEVICE
pc = ParameterContainer(
    in_features=IN_FEATURES, 
    out_features=OUT_FEATURES, 
    radius=RADIUS,
    beta=float(args.beta), 
    PNet_hidden_neurons=PNET_HIDDEN_NEURONS,
    n_actions=N_ACTIONS,
    epochs=EPOCHS,
    lr=LR,
    seq_len=SEQ_LEN,
    accumulation_steps=ACCUMULATION_STEPS,
    switch_time=SWITCH_TIME,
    eval_freq=EVAL_FREQ
)
encoder = LSTM_Encoder(
    in_features=pc.in_features,
    out_features=pc.out_features,
    n_states=pc.out_features,
    n_hidden=pc.PNet_hidden_neurons,
    n_actions=pc.n_actions,
    batch_size=1
).to(device)
decoder = Decoder(  
    in_features=pc.PNet_hidden_neurons*2, 
    out_features=pc.in_features
).to(device)
policy = policyNet( 
    hidden_size=pc.PNet_hidden_neurons,
    embedding_length=pc.out_features,
).to(device)
# encoder = par.DistributedDataParallel(encoder)
# rnn_policy_network = par.DistributedDataParallel(rnn_policy_network)
# decoder = par.DistributedDataParallel(decoder)
opt = torch.optim.Adam([{'params': encoder.parameters()},
                        {'params': decoder.parameters()}], lr=pc.lr)
policy_opt = torch.optim.Adam([{'params': policy.parameters()}], lr=pc.lr)
base_log_folder = './w2v_results/' + os.path.basename(__file__)
base_name = 'w2v_{}_{}_{}_{}'.format(
    pc.epochs, pc.out_features, pc.radius, pc.beta
)
if not os.path.exists(base_log_folder):
    os.mkdir(base_log_folder)
log_file = base_log_folder + base_name + '.json'
recorder = Recorder(
    path=log_file,
    hyper_parameters=pc.parameters,
    arg_dicts=[
        {'name': 'epoch', 'max_value': pc.epochs},
    ]
)
dataset = Dataset(dataset_name='x86',
                  csv_file='dataset/x86/x86_samples.csv',
                  seq_len=pc.seq_len,
                  fmt=['category', 'name', 'length', 'seq'])
dataset.initialize_word2index('./dataset/word2index.json')
dataset.set_vector_type('one-hot')
dataset.set_mode(mode='all')
w2v_dataset = W2VWrapper(dataset=dataset, radius=pc.radius)
dataloader = DataLoader(w2v_dataset, batch_size=1, shuffle=False, num_workers=0)
loss_func = CrossEntropyLoss()
reward_func = Reward()
rewards, losses, actions = [], [], []
agent = PPO.agent.PPO(state_dim=pc.PNet_hidden_neurons * 2 + pc.out_features, action_dim=2, )
if __name__ == "__main__":
    dataList = []
    # dataList = torch.load(base_log_folder + '/dataList.pt', map_location=device)['dataList']
    start_time = time.time()
    for epoch in range(1, 2):
        for i_episode, (data, label, target) in enumerate(dataloader):
            time_step = (i_episode + 1)

            data = data[0].contiguous().to(device)
            label = label.view(-1).contiguous().to(device)  
            code = encoder(data).to(device)  
            target_code = torch.squeeze(encoder(target.to(device)).to(device), dim=0)

            row, col = code.shape  
            hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            for i in range(int(row / 2)): 
                hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device), (hx, cx))
            hid_left = hx
            hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            for i in reversed(range(int(row / 2), row)): 
                hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device),
                                                 (hx, cx))
            hid_right = hx

            predict = decoder(
                torch.cat([hid_left, hid_right], dim=1)) 

            loss = loss_func(predict, label) / pc.accumulation_steps
            loss.backward()
            opt.step()
            opt.zero_grad()

            if loss >= 0.1:
                dataList.append([time_step, code.detach(), label, target_code.detach(), loss.detach()])

            losses.append(loss.detach().cpu().item())
            print('\rEpoch:[{}/{}] Step:{:5}. Loss:{:.3f}. DataListNums:{}.'.
                  format(epoch+1, 10, time_step, loss, len(dataList)), end='')
        # if (epoch+1) % 100 == 0:
        #     torch.save(
        #         obj={
        #             'encoder': encoder.state_dict(),
        #             'decoder': decoder.state_dict(),
        #             'opt': opt.state_dict(),
        #         },
        #         f=base_log_folder + "/first({})_{}".format(pc.beta, epoch+1) + '.pt'
        #     )
    # torch.save(
    #     obj={
    #         # 'code': code, 'label': label, 'target_code': target_code, 'loss': loss,
    #         'dataList': dataList
    #     },
    #     f=base_log_folder + "/dataList" + '.pt'
    # )

    testlength = len(dataList)
    testepochfreq = 10
    s = 0
    for i in range(testlength):
        s += dataList[i][-1]
    s /= testlength
    # print("\nOrigin Loss", s.item())
    end_time = time.time()
    print("time cost:", end_time - start_time, "s")
    for epoch in range(1, 2):
        for i_episode, (target_time_step, code, label, target_code, target_loss) in enumerate(dataList[:testlength]):
            time_step = (i_episode + 1)

            row, col = code.shape 
            state = torch.zeros(row, pc.PNet_hidden_neurons * 2 + pc.out_features).to(device)
            action = torch.zeros(row)
            prob = torch.zeros(row)
            val = torch.zeros(row)
            hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            hid = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            T = 0
            for i in range(int(row / 2)):
                state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
                action[i], prob[i], val[i] = agent.choose_action(state[i])
                if action[i] == 1:
                    T += 1
                    hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device), (hx, cx))  # LSTM层
            hid_left = hx
            hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
            for i in reversed(range(int(row / 2), row)):
                state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
                action[i], prob[i], val[i] = agent.choose_action(state[i])
                if action[i] == 1:
                    T += 1
                    hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device), (hx, cx))  # LSTM层
            hid_right = hx
            predict = decoder(torch.cat([hid_left, hid_right], dim=1))

            loss = loss_func(predict, label) / pc.accumulation_steps

            reward = (target_loss-loss+pc.beta*(row-T)/row)*1000
            for i in range(int(row/2)):
                agent.memory.push(state[i].detach().cpu().numpy(), action[i].detach().cpu(), prob[i].detach().cpu(), val[i].detach().cpu(),
                                  reward.detach().cpu()*i/row*2, True if i == row/2-1 else False)
            for i in range(int(row/2), row):
                agent.memory.push(state[i].detach().cpu().numpy(), action[i].detach().cpu(), prob[i].detach().cpu(), val[i].detach().cpu(),
                                  reward.detach().cpu()*(row-i)/row*2, True if i == row/2 else False)

            losses.append(loss.detach().cpu().item())
            rewards.append(reward.detach().cpu().item())
            actions.append(action.detach().cpu())

            print('\rEpoch:{}/{} Step:{:5}. Rew:{:.3f}. Act:{}. ML:{:.3f}'.
                  format(epoch, 1000, time_step, np.mean(rewards[-testlength*testepochfreq:]), action, np.mean(losses[-testlength*testepochfreq:])), end='')

        agent.update()
        agent.memory.clear()
        if (epoch % testepochfreq) == 0:
            print('')

    end_time = time.time()
    print("time cost:", end_time - start_time, "s")

    for epoch in range(1, 2):
        dataList_index = 0
        for i_episode, (data, label, target) in enumerate(dataloader):
            time_step = (i_episode + 1)
            if dataList_index < len(dataList) and time_step == dataList[dataList_index][0]:
                target_time_step, code, label, target_code, target_loss = dataList[dataList_index]
                dataList_index += 1
                row, col = code.shape 
                state = torch.zeros(row, pc.PNet_hidden_neurons * 2 + pc.out_features).to(device)
                action = torch.zeros(row)
                prob = torch.zeros(row)
                val = torch.zeros(row)
                hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                hid = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                T = 0
                for i in range(int(row / 2)):
                    state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
                    action[i], prob[i], val[i] = agent.choose_action(state[i])
                    if action[i] == 1:
                        T += 1
                        hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device),
                                                        (hx, cx)) 
                hid_left = hx
                hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                for i in reversed(range(int(row / 2), row)):
                    state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
                    action[i], prob[i], val[i] = agent.choose_action(state[i])
                    if action[i] == 1:
                        T += 1
                        hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device),
                                                         (hx, cx)) 
                hid_right = hx
                predict = decoder(torch.cat([hid_left, hid_right], dim=1))

                loss = loss_func(predict, label) / pc.accumulation_steps
                loss.backward()
                opt.step()
                opt.zero_grad()

                reward = (target_loss - loss + pc.beta * (row - T) / row) * 100
                for i in range(int(row / 2)):
                    agent.memory.push(state[i].detach().cpu().numpy(), action[i].detach().cpu(),
                                      prob[i].detach().cpu(), val[i].detach().cpu(),
                                      reward.detach().cpu() * i / row * 2, True if i == row / 2 - 1 else False)
                for i in range(int(row / 2), row):
                    agent.memory.push(state[i].detach().cpu().numpy(), action[i].detach().cpu(),
                                      prob[i].detach().cpu(), val[i].detach().cpu(),
                                      reward.detach().cpu() * (row - i) / row * 2, True if i == row / 2 else False)

                losses.append(loss.detach().cpu().item())
                rewards.append(reward.detach().cpu().item())
                actions.append(action.detach().cpu())

                print('\rEpoch:{}/{} Step:{:5}. Rew:{:.3f}. Act:{}. ML:{:.3f}'.
                      format(epoch + 1, 10, time_step, np.mean(rewards[-testlength * testepochfreq:]), action,
                             np.mean(losses[-testlength * testepochfreq:])), end='')

                agent.update()
                agent.memory.clear()
            else:
                data = data[0].contiguous().to(device) 
                label = label.view(-1).contiguous().to(device)  
                code = encoder(data).to(device)  
                target_code = torch.squeeze(encoder(target.to(device)).to(device), dim=0)

                row, col = code.shape  
                hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                for i in range(int(row / 2)):  
                    hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device), (hx, cx))
                hid_left = hx
                hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
                for i in reversed(range(int(row / 2), row)): 
                    hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device),
                                                     (hx, cx))
                hid_right = hx

                predict = decoder(
                    torch.cat([hid_left, hid_right], dim=1)) 

                loss = loss_func(predict, label) / pc.accumulation_steps
                loss.backward()
                opt.step()
                opt.zero_grad()

                losses.append(loss.detach().cpu().item())
                print('\rEpoch:[{}/{}] Step:{:5}. Loss:{:.3f}. '.
                      format(epoch + 1, 10, time_step, loss), end='')

    end_time = time.time()
    print("time cost:", end_time - start_time, "s")
    # recorder.save_numpy(directory=base_log_folder + "/PPO_{}/{}".format(pc.beta, base_name), filename="losses", numpy_obj=losses)
    # recorder.save_numpy(directory='./w2v_results/ppo', filename="rewards", numpy_obj=rewards)
    # recorder.save_numpy(directory='./w2v_results/ppo', filename="actions", numpy_obj=actions)
    # torch.save(
    #     obj={
    #         'decoder': decoder.state_dict(),
    #     },
    #     f=base_log_folder + "/PPO_{}/{}".format(pc.beta, base_name) + '.pt'
    # )
    # pickle.dump(
    #     obj=decoder.cpu().fc.weight.data.tolist(),
    #     file=open(
    #         base_log_folder + "/PPO_{}/{}".format(pc.beta, base_name) + '.pickle', 'wb'
    #     )
    # )

dataset.set_mode(mode='test_set')
w2v_dataset = W2VWrapper(dataset=dataset, radius=pc.radius) 
dataloader = DataLoader(w2v_dataset, batch_size=1, shuffle=False, num_workers=0)
start_time = time.time()
for i_episode, (data, label, target) in enumerate(dataloader):
    time_step = (i_episode + 1)
    if dataList_index < len(dataList) and time_step == dataList[dataList_index][0]: 
        target_time_step, code, label, target_code, target_loss = dataList[dataList_index]
        dataList_index += 1
        row, col = code.shape 
        state = torch.zeros(row, pc.PNet_hidden_neurons * 2 + pc.out_features).to(device)
        action = torch.zeros(row)
        prob = torch.zeros(row)
        val = torch.zeros(row)
        hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        hid = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        T = 0
        for i in range(int(row / 2)):
            state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
            action[i], prob[i], val[i] = agent.choose_action(state[i])
            if action[i] == 1:
                T += 1
                hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device),
                                                (hx, cx))  
        hid_left = hx
        hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        for i in reversed(range(int(row / 2), row)):
            state[i] = torch.cat([cx, hx, torch.unsqueeze(code[i], dim=0)], dim=1)
            action[i], prob[i], val[i] = agent.choose_action(state[i])
            if action[i] == 1:
                T += 1
                hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device),
                                                 (hx, cx))  
        hid_right = hx
        predict = decoder(torch.cat([hid_left, hid_right], dim=1))
    else:
        data = data[0].contiguous().to(device)  
        label = label.view(-1).contiguous().to(device)  
        code = encoder(data).to(device) 
        target_code = torch.squeeze(encoder(target.to(device)).to(device), dim=0)

        row, col = code.shape  
        hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        for i in range(int(row / 2)):  
            hx, cx = encoder.lstm_cell_left(torch.unsqueeze(code[i], dim=0).to(device), (hx, cx))
        hid_left = hx
        hx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        cx = torch.zeros(encoder.batch_size, encoder.n_hidden).to(device)
        for i in reversed(range(int(row / 2), row)): 
            hx, cx = encoder.lstm_cell_right(torch.unsqueeze(code[i], dim=0).to(device),
                                             (hx, cx))
        hid_right = hx

        predict = decoder(
            torch.cat([hid_left, hid_right], dim=1)) 
end_time = time.time()
print("time cost:", end_time - start_time, "s")
