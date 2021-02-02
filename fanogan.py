from torch.utils.data import DataLoader
from torch import optim
from torch import autograd
from torch import nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import sampler
from argparse import ArgumentParser
from wgan64x64 import *
from sklearn import metrics
import torch
import numpy as np
import time
import os
import sys
import random
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!

MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
DIM = 64  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 100000  # How many generator iterations to train for
OUTPUT_DIM = 3 * 64 * 64  # Number of pixels in image (3*64*64)
NOISE_SIZE = 128
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(
        real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def one_class_dataloader(c, nw=0, bs=64):
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    cifar = datasets.CIFAR10('./', download=False,
                             train=True, transform=transform)
    labels = np.array(cifar.targets)
    class_indices = np.argwhere(labels == c)
    class_indices = class_indices.reshape(class_indices.shape[0])
    trainloader = DataLoader(
        cifar, bs, sampler=sampler.SubsetRandomSampler(class_indices),
        num_workers=nw, pin_memory=True, drop_last=True)
    test = datasets.CIFAR10('./', download=False,
                           train=False, transform=transform)
    testloader = DataLoader(test, bs*2, num_workers=nw, pin_memory=True)

    return trainloader, testloader


def wgan_training():
    netG = GoodGenerator().to(device)
    netD = GoodDiscriminator().to(device)

    one = torch.FloatTensor([1]).to(device)
    mone = one * -1

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.0, 0.9))

    dataloader, _ = one_class_dataloader(options.c, 2, BATCH_SIZE)
    D_real_list = []
    D_fake_list = []
    D_cost_list = []
    G_cost_list = []
    for iteration in range(1, ITERS + 1):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for i, (_data, _) in enumerate(dataloader):
            if i == CRITIC_ITERS:
                break
            netD.zero_grad()

            # train with real

            real_data = _data.to(device)

            # import torchvision
            # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
            # torchvision.utils.save_image(real_data, filename)

            D_real = netD(real_data)
            D_real = D_real.mean()
            D_real.backward(mone)
            D_real_list.append(D_real.item())

            # train with fake
            noise = torch.randn(BATCH_SIZE, NOISE_SIZE)
            noise = noise.to(device)
            fake = netG(noise).detach()
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            D_fake_list.append(D_fake.item())

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                netD, real_data.data, fake.data)
            gradient_penalty.backward()

            # print "gradien_penalty: ", gradient_penalty

            D_cost = D_fake - D_real + gradient_penalty
            D_cost_list.append(D_cost.item())
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, 128)
        noise = noise.to(device)
        fake = netG(noise)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
        G_cost_list.append(G_cost.item())

        # Write logs and save samples
        if iteration % 20 == 0:
            save_image(fake*0.5+0.5, 'wgangp/{}.jpg'.format(iteration))
            print('Iters:{}, D(real):{}, D(fake):{}, Loss D:{}, Loss G:{}'.format(
                iteration,
                np.mean(D_real_list),
                np.mean(D_fake_list),
                np.mean(D_cost_list),
                np.mean(G_cost_list),)
            )
        if iteration % 1000 == 0 and iteration != 0:
            torch.save(netD.state_dict(), 'wgangp/netD_%d.pth' % iteration)
            torch.save(netG.state_dict(), 'wgangp/netG_%d.pth' % iteration)


def train_encoder():
    netG = GoodGenerator().to(device)
    netG.load_state_dict(torch.load('wgangp/netG_100000.pth'))
    netG.eval()
    netD = GoodDiscriminator().to(device)
    netD.load_state_dict(torch.load('wgangp/netD_100000.pth'))
    netD.eval()
    for p in netD.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = False

    dataloader, _ = one_class_dataloader(options.c, 2, BATCH_SIZE)

    netE = Encoder(DIM, NOISE_SIZE).to(device)
    # netE.load_state_dict(torch.load('wgangp/netE.pth'))

    optimizer = optim.Adam(netE.parameters(), 1e-4, (0.0, 0.9))

    crit = nn.MSELoss()

    for e in range(300):
        losses = []
        netE.train()
        for (x, _) in dataloader:
            x = x.to(device)
            code = netE(x)
            rec_image = netG(code)
            d_input = torch.cat((x, rec_image), dim=0)
            f_x, f_gx = netD.extract_feature(d_input).chunk(2, 0)
            loss = crit(rec_image, x) + options.alpha * crit(f_gx, f_x.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(e, np.mean(losses))
        netE.eval()
        rec_image = netG(netE(x))
        d_input = torch.cat((x, rec_image), dim=0)
        save_image(d_input*0.5+0.5, 'rec'+str(e)+'.bmp')
    torch.save(netE.state_dict(), 'wgangp/netE.pth')


def evaluate():
    netG = GoodGenerator().to(device)
    netG.load_state_dict(torch.load('wgangp/netG_100000.pth'))
    netG.eval()
    netD = GoodDiscriminator().to(device)
    netD.load_state_dict(torch.load('wgangp/netD_100000.pth'))
    netD.eval()
    netE = Encoder(DIM, NOISE_SIZE).to(device)
    netE.load_state_dict(torch.load('wgangp/netE.pth'))
    netE.eval()

    _, dataloader = one_class_dataloader(options.c, 0, BATCH_SIZE)
    # crit = nn.MSELoss()
    y_true, y_score = [], []
    in_real, out_real, in_rec, out_rec = [], [], [], []
    with torch.no_grad():
        for (x, label) in dataloader:
            bs = x.size(0)
            x = x.to(device)
            rec_image = netG(netE(x))
            d_input = torch.cat((x, rec_image), dim=0)
            idx = (label == options.c)
            in_real.append(x[idx])
            in_rec.append(rec_image[idx])
            idx = (label != options.c)
            out_real.append(x[idx])
            out_rec.append(rec_image[idx])
            f_x, f_gx = netD.extract_feature(d_input).chunk(2, 0)
            rec_diff = ((rec_image.view(bs, -1) - x.view(bs, -1))**2)
            rec_score = rec_diff.mean(dim=1) # - rec_diff.std(dim=1)
            feat_diff = ((f_x - f_gx)**2)
            feat_score = feat_diff.mean(dim=1) # + feat_diff.std(dim=1)
            outlier_score = rec_score + options.alpha * feat_score
            y_true.append(label)
            y_score.append(outlier_score.cpu())
    in_real = torch.cat(in_real, dim=0)[:32]
    in_rec = torch.cat(in_rec, dim=0)[:32]
    out_real = torch.cat(out_real, dim=0)[:32]
    out_rec = torch.cat(out_rec, dim=0)[:32]
    save_image(torch.cat((in_real, in_rec), dim=0), 'real.bmp', normalize=True)
    save_image(torch.cat((out_real, out_rec), dim=0),
               'fake.bmp', normalize=True)
    y_score = np.concatenate(y_score)
    y_true = np.concatenate(y_true)
    y_true[y_true != options.c] = -1
    y_true[y_true == options.c] = 1
    print('auc:', metrics.roc_auc_score(y_true, -y_score))
    # plt.figure()
    # plt.hist(y_score[y_true==1], 100, density=True, alpha=0.5, color='blue')
    # plt.hist(y_score[y_true==-1], 100, density=True, alpha=0.5, color='red')
    # plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--alpha', dest='alpha', type=float, default=1)
    parser.add_argument('--stage', dest='stage', type=int, default=1)
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--class', dest='c', type=int, required=True)
    parser.add_argument('--cuda', dest='cuda', type=str, default='0')
    global options
    options = parser.parse_args()
    device = torch.device('cuda:{}'.format(options.cuda))
    torch.cuda.set_device('cuda:{}'.format(options.cuda))
    if not options.eval:
        if options.stage == 1:
            wgan_training()
        elif options.stage == 2:
            train_encoder()
    else:
        evaluate()
