import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from utils import v_wrap, set_init

S_LEN = 5
A_DIM = 6

throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #actor
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.linear_2_a = nn.Linear(200, 100)
        self.output_a = nn.Linear(100, A_DIM)

        #critic
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.linear_2_c = nn.Linear(200, 100)
        self.output_c = nn.Linear(100, 1)

        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                  self.linear_1_c, self.linear_2_c, self.output_c])
        self.distribution = torch.distributions.Categorical


    def forward(self, x):
        linear_1_a = F.relu6(self.linear_1_a(x))
        linear_2_a = F.relu6(self.linear_2_a(linear_1_a))
        logits = self.output_a(linear_2_a)

        linear_1_c = F.relu6(self.linear_1_c(x))
        linear_2_c = F.relu6(self.linear_2_c(linear_1_c))
        values = self.output_c(linear_2_c)

        return logits, values


    def choose_action(self, mask, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits.view(1, -1), dim=1)
        m = self.distribution(prob)
        re = m.sample().numpy()[0]
        return np.int64(re), logits.data


    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values # advantage
        c_loss = td.pow(2) # value_loss = c_loss
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v # policy_loss = a_loss
        # entropy regularization ---
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(log_probs * probs).sum(1)
        a_loss -= 0.5 * entropy
        # entropy regularization ---
        total_loss = (c_loss + a_loss).mean()
        return total_loss


if __name__ == '__main__':

    # An instance of your model.
    lnet = Net()
    lnet.load_state_dict(torch.load('../../data/RL_model/2019-06-28_10-18-56/model/233293.pkl'))

    # An example input you would normally provide to your model's forward() method.

    data_list = [ 0.33333333,  0.33333333, -0.91149055,  0.00101571, -0.19378804]

    data = np.array(data_list)
    print(data_list)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(lnet, v_wrap(data[None, :]))
    traced_script_module.save("../../data/RL_model/2019-06-28_10-18-56/model.pt")