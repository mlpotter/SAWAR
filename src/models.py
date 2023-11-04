import torch
import torch.nn as nn
import torch.nn.functional as F

class Exponential_Model(nn.Module):
    def __init__(self,input_dim,hidden_layers=[10],output_dim=1):
        super().__init__()
        self.layers = [input_dim] + hidden_layers + [output_dim]
        self.linears = nn.ModuleList([nn.Linear(self.layers[i],self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.num_layers = len(self.layers)

    def rate(self,x):
        for i,l in enumerate(self.linears):
            x = l(x)

            if i == (self.num_layers-2):
                break

            x = F.leaky_relu(x)

        return torch.exp(x)

    def forward(self,x):

        rate = self.rate(x)
        k = torch.ones_like(rate)
        return rate,k


def main():
    input_dim = 5
    hidden_layers = [10,10]
    output_dim = 1

    batch_size = 10

    model = Exponential_Model(input_dim=input_dim,hidden_layers=hidden_layers,output_dim=output_dim)

    x = torch.randn(batch_size,input_dim)
    print(model)
    print(model.layers)
    rate,k = model(x)
    print(rate,k)


if __name__ == "__main__":
    main()

