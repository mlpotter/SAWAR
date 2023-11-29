import torch
from tqdm import tqdm

import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import random
import numpy as np
from copy import deepcopy
from torch import optim

from src.criterion import RightCensorWrapper,right_censored,ranking_loss

from csv import writer
from csv import reader


# TODO: customize for the right censored data analysis or exact time data analysis
def train(model,dataloader_train,optimizer,criterion,epochs,print_every=25,save_pth=None):
    train_loss = torch.zeros((epochs,))

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            xi,ti,yi = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            rate= model(xi)

            loss = criterion(rate,ti,yi)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if (epoch+1) % print_every == 0:
            print("Epoch {:d}, LL={:.3f}".format(epoch+1,running_loss))
        train_loss[epoch] = running_loss

    print('Finished Training')
    if save_pth is not None:
        torch.save(model.state_dict(),save_pth)

    return torch.arange(epochs),train_loss


def train_ranking(model, dataloader_train, optimizer, epochs, print_every=25, save_pth=None):
    train_loss = torch.zeros((epochs,))

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            xi, ti, yi = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            rate = model(xi)
            Ri = model.failure_cdf(xi,ti)

            loss = right_censored(rate, ti, yi) + ranking_loss(Ri,ti,yi)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if (epoch + 1) % print_every == 0:
            print("Epoch {:d}, LL={:.3f}".format(epoch + 1, running_loss))
        train_loss[epoch] = running_loss

    print('Finished Training')
    if save_pth is not None:
        torch.save(model.state_dict(), save_pth)

    return torch.arange(epochs), train_loss


# TODO: optimize min max?
def train_robust_step(model_loss, t, loader, eps_scheduler, norm, train, opt, bound_type, pareto=[0.5,0.5],method='robust',device="cpu"):
    meter = MultiAverageMeter()
    if train:
        model_loss.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model_loss.eval()
        eps_scheduler.eval()

    # model_loss.to(device)
    epoch_loss = 0
    for i, data in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        xi, ti, yi = data

        # xi = xi.to(device); ti = ti.to(device); yi = ti.to(device)

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-20:
            batch_method = "natural"
        if train:
            opt.zero_grad()

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        if norm > 0:
            ptb = PerturbationLpNorm(norm=norm, eps=eps)
        elif norm == 0:
            ptb = PerturbationL0Norm(eps=eps_scheduler.get_max_eps(),
                                     ratio=eps_scheduler.get_eps() / eps_scheduler.get_max_eps())

        # Make the input a BoundedTensor with the pre-defined perturbation.
        x_bounded = BoundedTensor(xi, ptb)


        regular_loss = model_loss(xi, ti, yi).sum()  # regular Right Censoring
        meter.update('Loss', regular_loss.item(), xi.size(0))

        if batch_method == "robust":
            # Compute LiRPA bounds using CROWN
            lb, ub = model_loss.compute_bounds(x=(x_bounded, ti, yi), IBP=True, method="backward", bound_upper=True,
                                               bound_lower=False)
            robust_loss = ub.sum()
            loss = robust_loss

        elif batch_method == "natural":
            loss = regular_loss

        combined_loss = ((pareto[0] * loss) + (pareto[1] * regular_loss))
        if train:
            combined_loss.backward()

            eps_scheduler.update_loss(loss.item() - regular_loss.item())
            opt.step()

        epoch_loss += combined_loss.detach().item()

        meter.update('Loss', loss.item(), xi.size(0))
        if batch_method != "natural":
            meter.update('Robust_Loss', robust_loss.item(), xi.size(0))

        meter.update('Time', time.time() - start)
        if i % 10 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

    return epoch_loss

def train_robust(model,dataloader_train,dataloader_val,method,args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # model = BoundedModule(clf, X_train)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.num_epochs/10), gamma=0.99)

    if method == "robust":
        scheduler_opts = args.scheduler_opts
    elif method == "natural":
        scheduler_opts = f"start={args.num_epochs+1},length={args.num_epochs+1}"

    eps_scheduler = eval(args.scheduler_name)(args.eps, scheduler_opts)



    timer = 0.0
    best_val_loss = np.inf
    best_epoch = 0
    window = []
    loss_train = np.zeros((args.num_epochs,))
    loss_val = np.zeros((args.num_epochs,))

    for t in range(1, args.num_epochs+1):
        if eps_scheduler.reached_max_eps():
            # Only decay learning rate after reaching the maximum eps
            lr_scheduler.step()
        print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
        start_time = time.time()
        # (model_loss, t, loader, eps_scheduler, norm, train, opt, bound_type, pareto=[0.5, 0.5], method='robust')
        train_epoch_loss = train_robust_step(model, t, dataloader_train, eps_scheduler, norm=args.norm, train=True, opt=optimizer,bound_type=args.bound_type,pareto=args.pareto,method=method,device=args.device)
        epoch_time = time.time() - start_time
        timer += epoch_time
        print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
        print("Evaluating...")
        with torch.no_grad():
            # how to clean up this...
            val_epoch_loss = train_robust_step(model, t, dataloader_val, eps_scheduler, norm=args.norm, train=False, opt=None, bound_type=args.bound_type,pareto=args.pareto,method=method,device=args.device)

            if len(window) < args.smooth_window:
                window.append(val_epoch_loss)
                val_epoch_smoothed = np.mean(window)
            else:
                window[:-1] = window[1:]
                window[-1] = val_epoch_loss
                val_epoch_smoothed = np.mean(window)

            if method == "robust":
                if (best_val_loss > val_epoch_smoothed) and (eps_scheduler.reached_max_eps()):
                    best_state_dict = deepcopy(model.state_dict())
                    best_val_loss = val_epoch_smoothed
                    best_epoch = t

            elif method == "natural":
                if (best_val_loss > val_epoch_smoothed):
                    best_state_dict = deepcopy(model.state_dict())
                    best_val_loss = val_epoch_smoothed
                    best_epoch = t

        if args.save_model != "":
            torch.save({'state_dict': model.state_dict(), 'epoch': t}, args.save_model)

        loss_train[t-1] = train_epoch_loss
        loss_val[t-1] = val_epoch_loss

    print(f"Best Validation Loss {best_val_loss} @ {best_epoch}")
    model.load_state_dict(best_state_dict)

    return np.arange(args.num_epochs),loss_train,loss_val

def lower_bound(clf, nominal_input, epsilon):
    # Wrap the model with auto_LiRPA.
    model = BoundedModule(clf, nominal_input)
    # Define perturbation. Here we add Linf perturbation to input data.
    ptb = PerturbationLpNorm(norm=np.inf, eps=torch.Tensor([epsilon]))
    # Make the input a BoundedTensor with the pre-defined perturbation.
    my_input = BoundedTensor(torch.Tensor(nominal_input), ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds using CROWN
    lb, ub = model.compute_bounds(x=(my_input,), method="backward")

    return lb, ub