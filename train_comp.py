import os
import copy
import argparse

import torch
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from scipy.stats import ortho_group
from sklearn import metrics
from torchsummary import summary

from data import AugMNISTDataset
from model import MLP, L0MLP


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--l1', type=float, default=4e-3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--name', type=str, default='compositionality')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=5e-4)
parser.add_argument('--rampup_begin', type=int, default=10)
parser.add_argument('--rampup_end', type=int, default=60)
parser.add_argument('--warmup_l1', type=float, default=1e-4)
parser.add_argument('--n_hidden', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--use_l0', type=int, default=0)
parser.add_argument('--use_l1', type=int, default=1)
parser.add_argument('--log_wandb', type=int, default=0)
parser.add_argument('--input_dim', type=int, default=13)
args = parser.parse_args()


if args.log_wandb:
    import wandb
    wandb.init(name=args.name,
               project=f"disentanglement")
    wandb.config.update(args)


torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
R = torch.tensor(ortho_group.rvs(dim=args.input_dim), dtype=torch.float)


def nonzero_params(model):
    total_params = 0
    nonzero = 0
    for p in model.parameters():
        total_params += p.numel()
        nonzero += (p.abs() > args.eps).sum()

    return nonzero, total_params

def generate_task():
    task = {}
    task['label'] = torch.tensor([[0,1,2,3,4]], dtype=torch.long)
    task['rotation_low'] = 0
    task['rotation_high'] = np.pi
    task['invert'] = False
    task['color_low'] = 20./255
    task['color_high'] = 80./255

    return task


def get_labels(samples, task):
    digits = (samples['label'].unsqueeze(1) == task['label']).any(-1)
    rotations = (samples['rotation'] >= task['rotation_low']) * (samples['rotation'] < task['rotation_high'])
    inversions = samples['invert'] == task['invert']
    colors = (samples['color'] > task['color_low']) * (samples['color'] < task['color_high'])

    return (digits * rotations * inversions * colors).float()


def get_features(samples: dict, entangle=False):
    N = samples['image'].shape[0]
    features = torch.zeros(samples['image'].shape[0], 10 + 1 + 1 + 1)
    features[torch.arange(N), samples['label']] = 1
    features[:,10] = samples['invert'].float()
    features[:,11] = samples['rotation']
    features[:,12] = samples['color']

    if entangle:
        features = features @ R.permute(1,0)
    return features


def get_split_configs():
    train_config = {
        'color': {
            **{idx: (0,60) for idx in range(0,10,2)},
            **{idx: (60,120) for idx in range(1,10,2)}
#            0: (0, 60),   1: (20, 80),   2: (40, 100), 3: (60, 120),
#            4: (80, 120), 5: (100, 120), 6: (0, 128),  7: (40, 60),
#            8: (20, 40),  9: (60, 100),
            }
    }

    test_config = {
        'color': {
            **{idx: (60,120) for idx in range(0,10,2)},
            **{idx: (0,60) for idx in range(1,10,2)}
#            0: (60, 128),   1: (80, 128),   2: (0, 40), 3: (0, 60),
#            4: (0, 80), 5: (0, 100), 6: (60, 128),  7: (40, 128),
#            8: (40, 100),  9: (0, 60),
        }
    }

    return train_config, test_config


def huber(p):
    eps = args.eps
    l1 = p[p.abs() > eps].abs() - eps/2
    l2 = p[p.abs() <= eps].pow(2) / (eps*2)
    return l1.sum() + l2.sum()


def l1(model):
    return sum([p.abs().sum() + 0.1 * p.pow(2).sum() for p in model.parameters()])
    #return sum([huber(p) for p in model.parameters()])


def copy_and_zero(model):
    model_ = copy.deepcopy(model)
    for p in model_.parameters():
        p[p.abs()<args.eps] = 0
    return model_

def print_stats(stats):

    for key in stats.keys():
        value = stats[key]
        print(f'{key} := {value}')

def run():
    print(f'Running from {os.getcwd()}')
    train_config, val_config = get_split_configs()
    print(f'Running with\n\ttrain_config: {train_config}\n\tval_config: {val_config}')

    train = AugMNISTDataset(transforms=['color'], config=train_config)
    val = AugMNISTDataset(transforms=['color'], config=val_config)
    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=1000, num_workers=0)

    mlp_width = 512
    if args.use_l0:
        e_model = L0MLP(args.n_hidden, args.input_dim, mlp_width, 1).to(args.device)
        d_model = L0MLP(args.n_hidden, args.input_dim, mlp_width, 1).to(args.device)
    else:
        e_model = MLP(args.n_hidden, args.input_dim, mlp_width, 1).to(args.device)
        d_model = MLP(args.n_hidden, args.input_dim, mlp_width, 1).to(args.device)


    #summary(e_model, (13,))
    #summary(d_model, (13,))

    if args.optimizer == 'sgd':
        e_opt = torch.optim.SGD(e_model.parameters(), momentum=0.9, lr=args.lr)
        d_opt = torch.optim.SGD(d_model.parameters(), momentum=0.9, lr=args.lr)
    elif args.optimizer == 'adam':
        e_opt = torch.optim.Adam(e_model.parameters(), lr=args.lr)
        d_opt = torch.optim.Adam(d_model.parameters(), lr=args.lr)
    step = 0
    task = generate_task()
    decay_epochs = [60,90,120,150]
    e_sched = torch.optim.lr_scheduler.MultiStepLR(e_opt, milestones=decay_epochs, gamma=0.1)
    d_sched = torch.optim.lr_scheduler.MultiStepLR(d_opt, milestones=decay_epochs, gamma=0.1)
    for epoch in range(args.epochs):
        for idx, samples in enumerate(train_dataloader):
            features = get_features(samples).to(args.device)
            entangled_features = get_features(samples, entangle=True).to(args.device)
            labels = get_labels(samples, task).to(args.device)

            if args.use_l0:
                e_out, l0_e  = e_model(entangled_features)
                d_out, l0_d = d_model(features)
            else:
                e_out = e_model(entangled_features)
                d_out = d_model(features)

            e_pred = e_out > 0
            e_acc = (e_pred == labels).float().mean()
            d_pred = d_out > 0
            d_acc = (d_pred == labels).float().mean()

            e_bce = F.binary_cross_entropy_with_logits(e_out, labels)
            e_loss = e_bce
            d_bce = F.binary_cross_entropy_with_logits(d_out, labels)
            d_loss = d_bce

            if args.use_l0:
                l0_coef = 1e-1
                d_loss += l0_coef * l0_d / len(samples)
                e_loss += l0_coef * l0_e / len(samples)

            if args.use_l1:
                if epoch <= args.rampup_begin:
                    l1_coef = args.warmup_l1
                else:
                    l1_coef = args.warmup_l1 + args.l1 / (args.warmup_l1 + args.l1) * min(args.l1, args.l1 * (float(epoch) - args.rampup_begin) / (args.rampup_end-args.rampup_begin))

                d_loss += l1_coef * l1(d_model)
                e_loss += l1_coef * l1(e_model)

            e_loss.backward()
            e_grad = torch.nn.utils.clip_grad_norm_(e_model.parameters(), 100)
            e_opt.step()
            e_opt.zero_grad()

            d_loss.backward()
            d_grad = torch.nn.utils.clip_grad_norm_(d_model.parameters(), 100)
            d_opt.step()
            d_opt.zero_grad()

            if step % 250 == 0:
                stats = {}
                stats['step'] = step
                stats['train_acc/e'], stats['train_acc/d']  = e_acc, d_acc
                stats['train_loss/e'], stats['train_loss/d']  = e_loss, d_loss
                stats['train_bce/e'], stats['train_bce/d']  = e_bce, d_bce

                if args.use_l1:
                    stats['l1_coef'] = l1_coef

                d_nonzero, d_params = nonzero_params(d_model)
                e_nonzero, e_params = nonzero_params(e_model)
                stats['d_nonzero'], stats['e_nonzero'] = d_nonzero, e_nonzero

                with torch.no_grad():
                    val_samples = next(iter(val_dataloader))
                    val_features = get_features(val_samples).to(args.device)
                    val_entangled_features = get_features(val_samples, entangle=True).to(args.device)
                    val_labels = get_labels(val_samples, task)

                    if args.use_l0:
                        e_out = copy_and_zero(e_model)(val_entangled_features)[0].cpu()
                        d_out = copy_and_zero(d_model)(val_features)[0].cpu()
                    else:
                        e_out = copy_and_zero(e_model)(val_entangled_features).cpu()
                        d_out = copy_and_zero(d_model)(val_features).cpu()

                    stats['val_auc/e'] = metrics.roc_auc_score(val_labels, e_out)
                    stats['val_auc/d'] = metrics.roc_auc_score(val_labels, d_out)
                    stats['lr/e'], stats['lr/d'] = e_sched.get_lr()[0], d_sched.get_lr()[0]

                    e_pred = e_out > 0
                    e_acc = (e_pred == val_labels).float().mean()
                    d_pred = d_out > 0
                    d_acc = (d_pred == val_labels).float().mean()

                    stats['val_acc/e'], stats['val_acc/d'] = e_acc, d_acc

                    # Fetch k wrong predictions
                    k = 10
                    e_wrong_mask = [e_pred != val_labels]
                    d_wrong_mask = [d_pred != val_labels]
                    wrong_preds_e, ftrs_e = e_out[e_wrong_mask][:k], val_entangled_features[:k]
                    wrong_preds_d, ftrs_d = d_out[d_wrong_mask][:k], val_features[:k]

                to_save = {
                    'd_model': d_model.state_dict(),
                    'e_model': e_model.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'e_opt': e_opt.state_dict()
                }
                torch.save(to_save, 'checkpoint.pt')
                if args.log_wandb:
                    wandb.log(stats)
                else:
                    print_stats(stats)

            step += 1
        e_sched.step()
        d_sched.step()

if __name__ == '__main__':
    run()
