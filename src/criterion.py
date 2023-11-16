import torch
import torch.nn as nn


def interval_censored(rate,t_start,t_end,event):
    pass

def right_censored(rate,t,event):
    log_exact = torch.log(rate) - (t*rate)
    log_right = -(rate*t)

    return (-event*log_exact - (1-event)*log_right).sum()

class RightCensorWrapper(nn.Module):
    def __init__(self,model):
        super(RightCensorWrapper,self).__init__()
        self.model = model

    def forward(self,x,t,e):
        rate = self.model(x)

        log_exact = e * torch.log(rate) + e * -(t * rate)
        log_right = (1 - e) * -(rate * t)

        return -log_exact + -log_right

def main():
    from src.models import Exponential_Model

    input_dim = 5
    hidden_layers = [10,10]
    output_dim = 1

    batch_size = 10
    x = torch.randn(batch_size,input_dim)

    model = Exponential_Model(input_dim=input_dim,hidden_layers=hidden_layers,output_dim=output_dim)

    rate,k = model.pdf_parameters(x)

    beta = torch.randn(input_dim,1)
    rate_true = torch.exp(x@beta)
    t_distribution = torch.distributions.exponential.Exponential(rate_true)
    t = t_distribution.sample()
    event = torch.ones_like(t)

    objective = right_censored

    print(objective(rate,k,t,event))

if __name__ == "__main__":
    main()

