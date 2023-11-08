import torch
from tqdm import tqdm

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
            lami,ki = model.pdf_parameters(xi)

            loss = criterion(lami,ki,ti,yi)
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