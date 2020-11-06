import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from data import AugMNISTDataset
import argparse
from model import VAE
from torchsummary import summary
from itertools import chain
import random


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--discrim_lr', type=float, default=3e-4)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--latent_dim', type=int, default=8)
parser.add_argument('--n_samples', type=int, default=6)
parser.add_argument('--n_sample_dims', type=int, default=None)
parser.add_argument('--reg_coef', type=float, default=None)
parser.add_argument('--name', type=str, default='disentanglement')
parser.add_argument('--n_show', type=int, default=64)
parser.add_argument('--discrim_hidden', type=int, default=100)
parser.add_argument('--discrim_hidden', type=int, default=100)
args = parser.parse_args()


def loss_function(recon_x, x, mu, logvar):
    MSE = (recon_x - x).pow(2).sum() / x.shape[0]

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return MSE, KLD


def save_samples(vae, z, fn):
    to_save = vae.decode(z).view(z.shape[0], 1, 28, 28)
    images = tv.utils.make_grid(to_save, int(z.shape[0]**0.5))
    tv.utils.save_image(images, fn)


def resample_kth_z(k, n, dim, device):
    z = torch.randn(int(n**0.5), 1, dim, device=device).repeat(1, int(n**0.5), 1).view(-1, dim)
    zk = torch.randn(n, device=device)
    z[:, k] = zk

    return z


def generate_subset_samples(batch_size, n_samples, n_sample_dims, dim, vae, device):
    z = torch.randn(1, batch_size, dim, device=device).repeat(n_samples, 1, 1)
    mask_ = []
    for batch_idx in range(batch_size):
        mask = (torch.randn(1, dim, device=device) > 0).repeat(n_samples, 1)
        mask_.append(mask[0])
        resample_values = torch.randn(n_samples * mask[0].sum(-1), device=device)
        z[:,batch_idx][mask] = resample_values

    samples = vae.decode(z)
    labels = torch.stack(mask_).float()
    return samples, labels, z


def run():
    dataset = AugMNISTDataset()
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=8)
    vae = VAE(latent_dim=args.latent_dim).to(args.device)
    discrim = nn.GRU(784, args.discrim_hidden, 1).to(args.device)
    discrim.flatten_parameters()
    classifier = nn.Sequential(nn.ReLU(), nn.Linear(args.discrim_hidden, args.latent_dim)).to(args.device)
    summary(vae, (784,))

    discrim_opt = torch.optim.Adam(chain(discrim.parameters(), classifier.parameters()), lr=args.discrim_lr)
    enc_opt = torch.optim.Adam(vae.e_params(), lr=args.lr)
    dec_opt = torch.optim.Adam(vae.d_params(), lr=args.lr)
    step = 0
    for epoch in range(args.epochs):
        for idx, sample in enumerate(dataloader):
            image = sample['image'].to(args.device).view(-1, 784)
            mu, logvar = vae.encode(image)
            z = vae.reparameterize(mu, logvar)
            x_hat = vae.decode(z)
            mse, kld = loss_function(x_hat, image, mu, logvar)
            vae_loss = mse + args.beta * kld

            vae_loss.backward()
            enc_grad = torch.nn.utils.clip_grad_norm(vae.e_params(), 100)
            enc_opt.step()
            enc_opt.zero_grad()

            if args.reg_coef:
                samples, labels, z = generate_subset_samples(args.batch_size//4, args.n_samples, args.n_sample_dims,
                                                             args.latent_dim, vae, args.device)
                discrim_out = discrim(samples.detach())[1][-1]
                logits = classifier(discrim_out)
                discrim_loss = F.binary_cross_entropy_with_logits(logits, labels)
                discrim_loss.backward()
                discrim_grad = torch.nn.utils.clip_grad_norm(chain(discrim.parameters(), classifier.parameters()), 100)
                discrim_opt.step()
                discrim_opt.zero_grad()

                dec_pre_grad = torch.nn.utils.clip_grad_norm(vae.d_params(), 100)

                discrim_out = discrim(samples)[1][-1]
                logits = classifier(discrim_out)
                discrim_loss = args.reg_coef * F.binary_cross_entropy_with_logits(logits, labels)
                discrim_loss.backward()
                discrim_opt.zero_grad()
                dec_post_grad = torch.nn.utils.clip_grad_norm(vae.d_params(), 100)
                dec_opt.step()
                dec_opt.zero_grad()
            else:
                dec_post_grad = torch.nn.utils.clip_grad_norm(vae.d_params(), 100)
                dec_opt.step()
                dec_opt.zero_grad()

            if step % 500 == 0:
                for k in range(args.latent_dim):
                    z = resample_kth_z(k, args.n_show, args.latent_dim, args.device)
                    save_samples(vae, z, f'z{k}_vae_output.png')

                z = torch.randn(args.n_show, args.latent_dim, device=args.device)
                save_samples(vae, z, 'vae_output.png')

                original_grid = tv.utils.make_grid(sample['image'][:args.n_show], int(args.n_show**0.5))
                tv.utils.save_image(original_grid, 'images.png')

                print(f'step := {step}')
                if args.reg_coef:
                    discrim_acc = ((logits>0) == labels).float().mean()
                    print(f'discrim_loss := {discrim_loss.item()}')
                    print(f'discrim_acc := {discrim_acc}')
                    print(f'grad/discrim := {discrim_grad}')
                    print(f'grad/dec_pre := {dec_pre_grad}')
                print(f'vae_loss := {vae_loss}')
                print(f'mse := {mse}')
                print(f'kld := {kld}')
                print(f'grad/enc := {enc_grad}')
                print(f'grad/dec_post := {dec_post_grad}')

            step += 1

if __name__ == '__main__':
    run()
