import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset



class TSLSTM(nn.Module):

    def __init__(self, input_dim = 1, hidden_dim = 32, n_layers = 1, n_out = 30, dropout = .5):
        super(TSLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_out = n_out

        self.lstm1 = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.n_layers,
            dropout = self.dropout,
            batch_first = True
        )

        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim*self.n_out)
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(self.hidden_dim*self.n_out, self.n_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        h_0, c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim, dtype=torch.float32).to(x.device)), Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim, dtype=torch.float32).to(x.device))
        out, (h_n, c_n) = self.lstm1(x, (h_0, c_0))

        h_n = h_n.view(-1, self.hidden_dim)
        output = self.lrelu(h_n)
        output = self.linear1(output)
        output = self.lrelu(output)
        output = self.drop_layer(output)
        output = self.linear2(output)

        return output



class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
