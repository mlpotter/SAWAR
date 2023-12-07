import torch
import torch.nn as nn
import math


def interval_censored(rate,t_start,t_end,event):
    pass

def right_censored(rate,t,event):
    log_exact = torch.log(rate) - (t*rate)
    log_right = -(rate*t)

    return (-event*log_exact - (1-event)*log_right).sum()

# def ranking_loss(rate,t,e):
#     # column vector
#     constant = math.log(2)
#     R = torch.transpose(rate, 0, 1) - rate
#
#     T = torch.relu(torch.sign(t.transpose(1, 0) - t))
#     # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
#
#     A = T * e
#     #  only remains T_{ij}=1 when event occured for subject i
#
#     # total pairs
#     N = torch.sum(A, axis=[], keepdims=True)
#
#     #
#     pairwise_ranking_loss = A * torch.log(torch.sigmoid(R)) / constant + A * torch.ones_like(rate)
#
#     # pairwise_ranking_loss.mean(axis=1, keepdim=True)
#     ci_lb = 1 / N * torch.sum(pairwise_ranking_loss, axis=1, keepdims=True)
#
#     return ci_lb.sum()
def ranking_loss(model,x,t,e,sigma=1):
    R = model.failure_cdf(x, t)
    # R{ij} = r_{i}{T_{j}} risk of the ith patient based on jth time condition
    # R{ij}' = r_{j}{T_{i}} risk of the jth patient based on the ith time condition

    Rii = 1.0 - torch.exp(-model(x) * t)
    # R{i} = R_{i}(T_{i})

    G = Rii - R.transpose(1, 0)
    # G_{ij} = r_{i}(T_{i}) - r_{j}(T_{i})

    T = torch.relu(torch.sign(t.transpose(1, 0) - t))
    # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

    A = T * e
    #  only remains T_{ij}=1 when event occured for subject i

    #
    pairwise_ranking_loss = A * torch.exp(-G / sigma)

    # pairwise_ranking_loss.mean(axis=1, keepdim=True)
    return torch.sum(pairwise_ranking_loss, axis=1, keepdims=True).sum()

class RightCensorWrapper(nn.Module):
    def __init__(self,model,**kwargs):
        super(RightCensorWrapper,self).__init__()
        self.model = model

    def forward(self,x,t,e):
        rate = self.model(x)

        log_exact = e * torch.log(rate) + e * -(t * rate)
        log_right = (1 - e) * -(rate * t)

        return -log_exact + -log_right


class RankingWrapper(nn.Module):
    def __init__(self, model,weight=1.0,sigma=1.0,**kwargs):
        super(RankingWrapper, self).__init__()
        self.model = model
        self.sigma = sigma
        self.weight = weight

    def forward(self, x, t, e):
        R = self.model.failure_cdf(x, t)
        # R{ij} = r_{i}{T_{j}} risk of the ith patient based on jth time condition
        # R{ij}' = r_{j}{T_{i}} risk of the jth patient based on the ith time condition

        Rii = 1.0 - torch.exp(-self.model(x) * t)
        # R{i} = R_{i}(T_{i})

        G = Rii - R.transpose(1, 0)
        # G_{ij} = r_{i}(T_{i}) - r_{j}(T_{i})

        T = torch.relu(torch.sign(t.transpose(1, 0) - t))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        A = T * e
        #  only remains T_{ij}=1 when event occured for subject i

        #
        pairwise_ranking_loss = A * torch.exp(-G / self.sigma)

        # pairwise_ranking_loss.mean(axis=1, keepdim=True)
        return self.weight*torch.sum(pairwise_ranking_loss, axis=1, keepdims=True)

# class RankingWrapper(nn.Module):
#     def __init__(self, model,**kwargs):
#         super(RankingWrapper, self).__init__()
#         self.model = model
#         self.constant = math.log(2)
#
#     def forward(self, x, t, e):
#         # column vector
#         mu = self.model(x)
#
#         R = torch.transpose(mu,0,1) - mu
#
#         T = torch.relu(torch.sign(t.transpose(1, 0) - t))
#         # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
#
#         A = T * e
#         #  only remains T_{ij}=1 when event occured for subject i
#
#         # total pairs
#         N = torch.sum(A,axis=[],keepdims=True)
#
#         #
#         pairwise_ranking_loss = A * torch.log(torch.sigmoid(R))/self.constant + A * torch.ones_like(mu)
#
#         # pairwise_ranking_loss.mean(axis=1, keepdim=True)
#         ci_lb = 1/N * torch.sum(pairwise_ranking_loss, axis=1, keepdims=True)
#
#         return ci_lb

class RHC_Ranking_Wrapper(nn.Module):
    def __init__(self, model,weight=1.0,sigma=1.0,**kwargs):
        super(RHC_Ranking_Wrapper, self).__init__()
        self.model = model
        self.weight = weight
        self.sigma = sigma

    def forward(self,x,t,e):
        return self.weight*self.Rank(x,t,e) + self.RHC(x,t,e)

    def Rank(self,x,t,e):
        R = self.model.failure_cdf(x, t)
        # R{ij} = r_{i}{T_{j}} risk of the ith patient based on jth time condition
        # R{ij}' = r_{j}{T_{i}} risk of the jth patient based on the ith time condition

        Rii = 1.0 - torch.exp(-self.model(x) * t)
        # R{i} = R_{i}(T_{i})

        G = Rii - R.transpose(1, 0)
        # G_{ij} = r_{i}(T_{i}) - r_{j}(T_{i})

        T = torch.relu(torch.sign(t.transpose(1, 0) - t))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        A = T * e
        #  only remains T_{ij}=1 when event occured for subject i

        #
        pairwise_ranking_loss = A * torch.exp(-G / self.sigma)

        # pairwise_ranking_loss.mean(axis=1, keepdim=True)
        return torch.sum(pairwise_ranking_loss, axis=1, keepdims=True)
    def RHC(self,x,t,e):
        rate = self.model(x)

        log_exact = e * torch.log(rate) + e * -(t * rate)
        log_right = (1 - e) * -(rate * t)

        return -log_exact + -log_right

# class RHC_Ranking_Wrapper(nn.Module):
#     def __init__(self, model,weight=1.0,**kwargs):
#         super(RHC_Ranking_Wrapper, self).__init__()
#         self.model = model
#         self.weight = weight
#         self.constant = math.log(2)
#
#     def forward(self,x,t,e):
#         return self.weight*self.Rank(x,t,e) + self.RHC(x,t,e)
#
#     def Rank(self,x,t,e):
#         mu = self.model(x)
#
#         R = torch.transpose(mu,0,1) - mu
#
#         T = torch.relu(torch.sign(t.transpose(1, 0) - t))
#         # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
#
#         A = T * e
#         #  only remains T_{ij}=1 when event occured for subject i
#
#         # total pairs
#         N = torch.sum(A,axis=[],keepdims=True)
#
#         #
#         pairwise_ranking_loss = A * torch.log(torch.sigmoid(R))/self.constant + A * torch.ones_like(mu)
#
#         # pairwise_ranking_loss.mean(axis=1, keepdim=True)
#         ci_lb = 1/N * torch.sum(pairwise_ranking_loss, axis=1, keepdims=True)
#
#         return ci_lb
#     def RHC(self,x,t,e):
#         rate = self.model(x)
#
#         log_exact = e * torch.log(rate) + e * -(t * rate)
#         log_right = (1 - e) * -(rate * t)
#
#         return -log_exact + -log_right

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

    print(objective(rate,t,event))

    R = model.failure_cdf(x,t)

    rl = ranking_loss(R,t,event)
    print(rl)
    #
    # # print(t,event)
    print(torch.column_stack((rl,t,event)))

if __name__ == "__main__":
    main()

