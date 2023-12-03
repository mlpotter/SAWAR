import torch
import torch.nn as nn
import torch.nn.functional as F

class Exponential_Model(nn.Module):
    def __init__(self,input_dim,hidden_layers=[10],output_dim=1):
        super().__init__()
        self.layers = [input_dim] + hidden_layers + [output_dim]
        module_list = []
        for i in range(len(self.layers)-2):
            module_list.append(nn.Linear(self.layers[i],self.layers[i+1]))
            module_list.append(nn.LeakyReLU())

        module_list.append(nn.Linear(self.layers[i+1],self.layers[i+2]))

        # self.linears = nn.ModuleList([nn.Linear(self.layers[i],self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.module_list = nn.ModuleList(module_list)

        # self.num_layers = len(self.layers)

    def rate_logit(self,x):
        # for i,l in enumerate(self.linears):
        #     x = l(x)
        #
        #     if i == (self.num_layers-2):
        #         break
        #
        #     x = F.leaky_relu(x)

        for l in self.module_list:
            x = l(x)

        return x

    def forward(self,x):

        return torch.exp(self.rate_logit(x))

    def pdf_parameters(self,x):
        rate_logit = self.rate_logit(x)
        rate = torch.exp(rate_logit)
        k = torch.ones_like(rate)
        return rate,k

    def survival_qdf(self,x,t):

        t = t.view(1,-1)
        rate = self.forward(x)
        St = torch.exp( -(rate*t))

        return St

    def failure_cdf(self,x,t):

        t = t.view(1,-1)
        rate = self.forward(x)
        Ft = 1-torch.exp( -(rate*t))

        return Ft

class Weibull_Model(nn.Module):
    def __init__(self,input_dim,hidden_layers=[10],output_dim=1):
        super().__init__()
        self.layers = [input_dim] + hidden_layers + [output_dim]
        self.linears_rate = nn.ModuleList([nn.Linear(self.layers[i],self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.num_layers = len(self.layers)
        self.linears_k = nn.ModuleList([nn.Linear(self.layers[i],self.layers[i+1]) for i in range(len(self.layers)-1)])
        # self.k_logit = nn.Parameter(torch.FloatTensor([[0.0]]))
    def rate_logit(self,x):
        for i,l in enumerate(self.linears_rate):
            x = l(x)

            if i == (self.num_layers-2):
                break

            x = F.leaky_relu(x)

        return x

    def k_logit(self,x):
        for i,l in enumerate(self.linears_k):
            x = l(x)

            if i == (self.num_layers-2):
                break

            x = F.leaky_relu(x)

        return x

    def forward(self,x):
        rate_logits = self.rate_logit(x)
        # k_logit = self.k_logit(x)
        rate = torch.exp(rate_logits)


        k_logits = self.k_logit(x)

        k = torch.exp(k_logits)#*torch.ones_like(rate)

        # rate,k = self.pdf_parameters(x)
        return rate*k#torch.concat((rate,k),1)

    def pdf_parameters(self,x):
        rate_logit = self.rate_logit(x)
        # k_logit = self.k_logit(x)
        k = torch.exp(self.k_logit)*torch.ones_like(rate_logit)
        rate = torch.exp(rate_logit)
        # k = torch.exp(k_logit)
        return rate,k

    def survival_qdf(self,x,t):

        t = t.view(1,-1)
        parameters = self.forward(x)
        rate,k = parameters[:,[0]],parameters[:,[1]]
        St = torch.exp( -torch.pow(rate,k)*torch.pow(t,k))

        return St

    def failure_cdf(self,x,t):

        t = t.view(1,-1)
        rate,k = self.forward(x)
        Ft = 1. - torch.exp( -torch.pow(rate,k)*torch.pow(t,k))

        return Ft

def main():
    input_dim = 5
    hidden_layers = [10]
    output_dim = 1

    batch_size = 10

    # Exponential model
    model = Exponential_Model(input_dim=input_dim,hidden_layers=hidden_layers,output_dim=output_dim)

    x = torch.randn(batch_size,input_dim)
    print(model)
    print(model.layers)
    rate,k = model.pdf_parameters(x)
    print(rate)

    t = torch.linspace(0,5,15)

    St = model.survival_qdf(x,t)
    print(St.shape)

    model = Weibull_Model(input_dim=input_dim,hidden_layers=hidden_layers,output_dim=output_dim)
    ratek = model(x)

    print(rate)

if __name__ == "__main__":
    main()

