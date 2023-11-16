import torch
from tqdm import tqdm

import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import random
import numpy as np
from torch import optim

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

# TODO: optimize min max?
def train_robust_step(model_loss, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust'):
    meter = MultiAverageMeter()
    if train:
        model_loss.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model_loss.eval()
        eps_scheduler.eval()

    for i, data in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        xi, ti, yi = data

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
            lb, ub = model_loss.compute_bounds(x=(x_bounded, ti, yi), IBP=False, method="backward", bound_upper=True,
                                               bound_lower=False)
            robust_loss = ub.sum()
            loss = robust_loss
        elif batch_method == "natural":
            loss = regular_loss

        if train:
            (.1 * loss + 0.9 * regular_loss).backward()
            eps_scheduler.update_loss(loss.item() - regular_loss.item())
            opt.step()
        meter.update('Loss', loss.item(), xi.size(0))
        if batch_method != "natural":
            meter.update('Robust_Loss', robust_loss.item(), xi.size(0))

        meter.update('Time', time.time() - start)
        if i % 10 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

def train_robust(model,dataloader_train,dataloader_test,method,args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)

    if method == "robust":
        scheduler_opts = args.scheduler_opts
    elif method == "natural":
        scheduler_opts = f"start={args.num_epochs+1},length={args.num_epochs+1}"

    eps_scheduler = eval(args.scheduler_name)(args.eps, scheduler_opts)



    timer = 0.0
    for t in range(1, args.num_epochs+1):
        if eps_scheduler.reached_max_eps():
            # Only decay learning rate after reaching the maximum eps
            lr_scheduler.step()
        print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
        start_time = time.time()
        train_robust_step(model, t, dataloader_train, eps_scheduler, args.norm, True, optimizer, args.bound_type)
        epoch_time = time.time() - start_time
        timer += epoch_time
        print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
        print("Evaluating...")
        with torch.no_grad():
            train_robust_step(model, t, dataloader_test, eps_scheduler, norm, False, None, args.bound_type)

        if args.save_model != "":
            torch.save({'state_dict': model.state_dict(), 'epoch': t}, args.save_model)

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