import torch
import torch.nn as nn
import numpy as np
from config import *

device = DEVICE

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, batch_size):
        super().__init__()
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.neg_log_prob = None
        self.rnn_cell = nn.LSTMCell(n_states, n_hidden)  
        self.fc = nn.Linear(n_hidden*2+n_states, n_actions)
        self.rnn_cell.weight_ih.data.normal_(0, 0.1)
        self.rnn_cell.weight_hh.data.normal_(0, 0.1)
        self.fc.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, state): 
        return torch.log_softmax(self.fc(state), dim=1) 

    def hidden(self, code):
        self.neg_log_prob = 0

        row, col = code.shape 
        h = torch.zeros(self.batch_size, self.n_hidden).to(device) 
        hid = torch.zeros(self.batch_size, self.n_hidden).to(device)
        T = 1e-7  
        for i in range(row):
            temp = torch.unsqueeze(code[i], dim=0).to(device)
            h = self.rnn_cell(temp, h)

            log_prob = self.forward(h.detach()) 
            action = np.random.choice([0, 1, 2], p=np.exp(log_prob.detach().cpu().numpy())[0])  
            T += action  
            hid += action * h  

            self.neg_log_prob -= log_prob[0, action]  
        return hid / T


class RNNPolicyNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions, batch_size):
        super().__init__()
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.neg_log_prob = None
        self.rnn_cell = nn.RNNCell(n_states, n_hidden)
        self.fc = nn.Linear(n_hidden, n_actions)
        self.rnn_cell.weight_ih.data.normal_(0, 0.1)
        self.rnn_cell.weight_hh.data.normal_(0, 0.1)
        self.fc.weight.data.normal_(0, 0.1) 

    def forward(self, state): 
        return torch.log_softmax(self.fc(state), dim=1) 

    def hidden(self, code):
        self.neg_log_prob = 0

        row, col = code.shape 
        h = torch.zeros(self.batch_size, self.n_hidden).to(device)  
        hid = torch.zeros(self.batch_size, self.n_hidden).to(device)
        T = 1e-7  
        for i in range(row):
            temp = torch.unsqueeze(code[i], dim=0).to(device)
            h = self.rnn_cell(temp, h)

            log_prob = self.forward(h.detach())  
            action = np.random.choice([0, 1, 2], p=np.exp(log_prob.detach().cpu().numpy())[0]) 
            T += action  
            hid += action * h 

            self.neg_log_prob -= log_prob[0, action] 
        return hid / T


class policyNet(nn.Module):
    def __init__(self, hidden_size, embedding_length):
        super(policyNet, self).__init__()
        self.hidden = hidden_size
        self.embedding = embedding_length
        self.W1 = nn.Parameter(torch.cuda.FloatTensor(self.hidden*2, 1).uniform_(-0.5, 0.5))
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(self.embedding, 1).uniform_(-0.5, 0.5))
        self.b = nn.Parameter(torch.cuda.FloatTensor(1, 1).uniform_(-0.5, 0.5))
        # self.W1 = nn.Parameter(torch.nn.init.kaiming_uniform(torch.cuda.FloatTensor(self.hidden * 2, 1), a=0, mode='fan_in'))
        # self.W2 = nn.Parameter(torch.nn.init.kaiming_uniform(torch.cuda.FloatTensor(self.embedding, 1), a=0, mode='fan_in'))
        # self.b = nn.Parameter(torch.nn.init.kaiming_uniform(torch.cuda.FloatTensor(1, 1), a=0, mode='fan_in'))

    def forward(self, state):
        h, x = state[0, :self.hidden*2], state[0, self.hidden*2:]
        h_ = torch.matmul(h.view(1, -1), self.W1) 
        x_ = torch.matmul(x.view(1, -1), self.W2) 
        scaled_out = torch.sigmoid(h_ + x_ + self.b) 
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out], 0)
        return torch.log(scaled_out).view(-1)

    def get_gradient(self, state):
        log_prob = self.forward(state) 
        action = np.random.choice([0, 1], p=np.exp(log_prob.detach().cpu().numpy())) 
        grad = torch.autograd.grad(log_prob[int(action)].view(-1),
                                   self.parameters())
        return action, grad


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.neg_log_prob = 0
        # self.b = nn.Parameter(torch.FloatTensor(1, 1).uniform_(-0.5, 0.5))
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, state): 
        x = torch.relu(self.fc1(state))
        return torch.log_softmax(self.fc2(x), dim=1) 

    def hidden(self, code):
        self.neg_log_prob = 0

        row, col = code.shape  
        h = torch.zeros(1, col).to(device) 
        #print(self.n_states,self.n_hidden,self.n_actions,h.shape,code.shape)
        T = 1e-6  
        for i in range(row):
            state = torch.cat([h, torch.unsqueeze(code[i], dim=0)], dim=1)

            log_prob = self.forward(state.detach()) 
            action = np.random.choice([0, 1, 2], p=np.exp(log_prob.detach().cpu().numpy())[0]) 

            T += action 
            h += action * torch.unsqueeze(code[i], dim=0)  

            self.neg_log_prob -= log_prob[0, action] 
        return h/T


class LSTM_Encoder(nn.Module):
    def __init__(self, in_features, out_features, n_states, n_hidden, n_actions, batch_size):
        super(LSTM_Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)  
        self.fc.weight.data.normal_(0, 0.1) 

        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.neg_log_prob = None

        self.lstm_cell_left = nn.LSTMCell(n_states, n_hidden) 
        self.lstm_cell_left.weight_ih.data.normal_(0, 0.1)
        self.lstm_cell_left.weight_hh.data.normal_(0, 0.1)
        self.lstm_cell_right = nn.LSTMCell(n_states, n_hidden) 
        self.lstm_cell_right.weight_ih.data.normal_(0, 0.1)
        self.lstm_cell_right.weight_hh.data.normal_(0, 0.1)

    def forward(self, x):
        code = self.fc(x)  
        return code
        ".detach().view(-1).cpu().numpy()"


class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False) 

    def forward(self, x):
        rows, cols = x.shape
        code = self.fc(x)  

        h = torch.zeros(1, self.out_features).to(device) 
        for i in range(rows):
            h += code[i]
        return h


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)  
        self.fc.weight.data.normal_(0, 0.1)

    def forward(self, h):
        return self.fc(h) 
        ".detach().view(-1).cpu().numpy()"


if __name__ == "__main__":
    rnn = RNNPolicyNetwork(n_states=128, n_hidden=32, n_actions=3, batch_size=1)
    xs = torch.randn(8, 128)
    ans = rnn.hidden(xs)
    print(ans.shape)
    print(ans)
