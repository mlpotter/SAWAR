import torch
import torch.nn as nn


def interval_censored(rate,t_start,t_end,event):
    pass

def right_censored(rate,t,event):
    log_exact = torch.log(rate) - (t*rate)
    log_right = -(rate*t)

    return (-event*log_exact - (1-event)*log_right).sum()

def ranking_loss(model,X_train,rate,t,event):
    print(rate)
    for ri,ti,ei in zip(rate,t,event):
        t[t>=ti]

    time_failed_first = t < t.T
    event_failed_first = event >= event.T

    model(X_train)

    R = model.survival_qdf(X_train,t)

    torch.diag(R)

    one_vector = torch.ones_like(t)

    R_ = one_vector @ torch.diag(R).reshape(1,-1)


    L = R_ - R
    L = L.T

    T = t < t.T


    I_2 = tf.cast(tf.equal(self.k, e + 1), dtype=tf.float32)  # indicator for event
    I_2 = tf.diag(tf.squeeze(I_2))
    tmp_e = tf.reshape(tf.slice(self.out, [0, e, 0], [-1, 1, -1]),
                       [-1, self.num_Category])  # event specific joint prob.

    R = tf.matmul(tmp_e, tf.transpose(self.fc_mask2))  # no need to divide by each individual dominator
    # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

    diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
    R = tf.matmul(one_vector, tf.transpose(diag_R)) - R  # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    R = tf.transpose(R)  # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

    T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
    # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

    T = tf.matmul(I_2, T)  # only remains T_{ij}=1 when event occured for subject i

    tmp_eta = tf.reduce_mean(T * tf.exp(-R / sigma1), reduction_indices=1, keep_dims=True)

    eta.append(tmp_eta)


    pass

    ### LOSS-FUNCTION 2 -- Ranking loss
    # def loss_Ranking(self):
    #     sigma1 = tf.constant(0.1, dtype=tf.float32)
    #
    #     eta = []
    #     for e in range(self.num_Event):
    #         one_vector = tf.ones_like(self.t, dtype=tf.float32)
    #         I_2 = tf.cast(tf.equal(self.k, e+1), dtype = tf.float32) #indicator for event
    #         I_2 = tf.diag(tf.squeeze(I_2))
    #         tmp_e = tf.reshape(tf.slice(self.out, [0, e, 0], [-1, 1, -1]), [-1, self.num_Category]) #event specific joint prob.
    #
    #         R = tf.matmul(tmp_e, tf.transpose(self.fc_mask3)) #no need to divide by each individual dominator
    #         # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})
    #
    #         diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
    #         R = tf.matmul(one_vector, tf.transpose(diag_R)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
    #         R = tf.transpose(R)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})
    #
    #         T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
    #         # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
    #
    #         T = tf.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i
    #
    #         tmp_eta = tf.reduce_mean(T * tf.exp(-R/sigma1), reduction_indices=1, keepdims=True)
    #
    #         eta.append(tmp_eta)
    #     eta = tf.stack(eta, axis=1) #stack referenced on subjects
    #     eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_Event]), reduction_indices=1, keepdims=True)
    #
    #     self.LOSS_2 = tf.reduce_sum(eta) #sum over num_Events
    # TODO: REVIEW
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

    print(objective(rate,t,event))

    print(ranking_loss(model,x,rate,t,event))

if __name__ == "__main__":
    main()

