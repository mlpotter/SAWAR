import torch
import torch.nn as nn


def interval_censored(rate,t_start,t_end,event):
    pass

def right_censored(rate,t,event):
    log_exact = torch.log(rate) - (t*rate)
    log_right = -(rate*t)

    return (-event*log_exact - (1-event)*log_right).sum()

def ranking_loss():

    # see ranking wrapper...

    # I_2 = tf.cast(tf.equal(self.k, e + 1), dtype=tf.float32)  # indicator for event
    # I_2 = tf.diag(tf.squeeze(I_2))
    # tmp_e = tf.reshape(tf.slice(self.out, [0, e, 0], [-1, 1, -1]),
    #                    [-1, self.num_Category])  # event specific joint prob.
    #
    # R = tf.matmul(tmp_e, tf.transpose(self.fc_mask2))  # no need to divide by each individual dominator
    # # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})
    #
    # diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
    # R = tf.matmul(one_vector, tf.transpose(diag_R)) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    # R = tf.transpose(R)  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})
    #
    # T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
    # # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
    #
    # T = tf.matmul(I_2, T)  # only remains T_{ij}=1 when event occured for subject i
    #
    # tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), reduction_indices=1, keep_dims=True)
    #
    # eta.append(tmp_eta)


    pass

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
    def __init__(self, model,sigma=1.0,**kwargs):
        super(RankingWrapper, self).__init__()
        self.model = model
        self.sigma = sigma

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
        return torch.mean(pairwise_ranking_loss, axis=1, keepdims=True)

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
        return torch.mean(pairwise_ranking_loss, axis=1, keepdims=True)
    def RHC(self,x,t,e):
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

    print(objective(rate,t,event))

    R = model.failure_cdf(x,t)

    rl = ranking_loss(R,t,event)
    print(rl)
    #
    # # print(t,event)
    print(torch.column_stack((rl,t,event)))

if __name__ == "__main__":
    main()

