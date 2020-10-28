import torch
import torchvision as tv
import torch.nn.functional as F
from data import AugMNISTDataset
import argparse
from model import MLP
import numpy as np
from scipy.stats import ortho_group
import os
import copy
from torchsummary import summary


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--l1', type=float, default=4e-3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--name', type=str, default='compositionality')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=5e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--warmup_start', type=int, default=5000)
parser.add_argument('--warmup_l1', type=float, default=1e-4)
parser.add_argument('--n_hidden', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
R = torch.tensor(ortho_group.rvs(dim=13), dtype=torch.float)


def nonzero_params(model):
    total_params = 0
    nonzero = 0
    for p in model.parameters():
        total_params += p.numel()
        nonzero += (p.abs() > args.eps).sum()

    return nonzero, total_params

def generate_task():
    task = {}
    task['label'] = torch.tensor([[0,2,4,7,8]], dtype=torch.long)
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
            0: (0, 60),   1: (20, 80),   2: (40, 100), 3: (60, 120),
            4: (80, 120), 5: (100, 120), 6: (0, 128),  7: (40, 60),
            8: (20, 40),  9: (60, 100),
        }
    }

    test_config = {
        'color': {
            0: (60, 128),   1: (80, 128),   2: (0, 40), 3: (0, 60),
            4: (0, 80), 5: (0, 100), 6: (60, 128),  7: (40, 128),
            8: (40, 100),  9: (0, 60),
        }
    }

    return train_config, test_config


def huber(p):
    l1 = p[p.abs() > args.eps].abs() - args.eps/2
    l2 = p[p.abs() <= args.eps].pow(2) / (args.eps*2)
    return l1.sum() + l2.sum()


def l1(model):
    #return sum([p.abs().sum() for p in model.parameters()])
    return sum([huber(p) for p in model.parameters()])


def copy_and_zero(model):
    model_ = copy.deepcopy(model)
    for p in model_.parameters():
        p[p.abs()<args.eps] = 0
    return model_


def run():
    print(f'Running from {os.getcwd()}')
    train_config, val_config = get_split_configs()
    print(f'Running with\n\ttrain_config: {train_config}\n\tval_config: {val_config}')

    train = AugMNISTDataset(transforms=['color'], config=train_config)
    val = AugMNISTDataset(transforms=['color'], config=val_config)
    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.batch_size, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=1000, num_workers=0)

    e_model = MLP(args.n_hidden, 10 + 1 + 1 + 1, 512, 1).to(args.device)
    d_model = MLP(args.n_hidden, 10 + 1 + 1 + 1, 512, 1).to(args.device)
    summary(e_model, (13,))
    summary(d_model, (13,))
    
    e_opt = torch.optim.SGD(e_model.parameters(), momentum=0.9, lr=args.lr)
    d_opt = torch.optim.SGD(d_model.parameters(), momentum=0.9, lr=args.lr)
    step = 0
    task = generate_task()
    e_sched = torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, args.epochs)
    d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, args.epochs)
    for epoch in range(args.epochs):
        for idx, samples in enumerate(train_dataloader):
            features = get_features(samples).to(args.device)
            entangled_features = get_features(samples, entangle=True).to(args.device)
            labels = get_labels(samples, task).to(args.device)
            e_out = e_model(entangled_features)
            d_out = d_model(features)

            e_pred = e_out > 0
            e_acc = (e_pred == labels).float().mean()
            d_pred = d_out > 0
            d_acc = (d_pred == labels).float().mean()

            if step <= args.warmup_start:
                l1_penalty = args.warmup_l1
            else:
                l1_penalty = args.warmup_l1 + args.l1 / (args.warmup_l1 + args.l1) * min(args.l1, args.l1 * (float(step) - args.warmup_start) / args.warmup_steps)

            e_bce = F.binary_cross_entropy_with_logits(e_out, labels)
            el1 = l1(e_model)
            e_loss = e_bce + l1_penalty * el1
            
            d_bce = F.binary_cross_entropy_with_logits(d_out, labels)
            dl1 = l1(d_model)
            d_loss = d_bce + l1_penalty * dl1

            e_loss.backward()
            e_grad = torch.nn.utils.clip_grad_norm_(e_model.parameters(), 100)
            e_opt.step()
            e_opt.zero_grad()

            d_loss.backward()
            d_grad = torch.nn.utils.clip_grad_norm_(d_model.parameters(), 100)
            d_opt.step()
            d_opt.zero_grad()

            if step % 250 == 0:
                print(f'step := {step}')
                print(f'l1_penalty := {l1_penalty}')
                print(f'train_acc/e := {e_acc}')
                print(f'train_acc/d := {d_acc}')
                print(f'l1/e := {el1}')
                print(f'l1/d := {dl1}')
                print(f'grad/e := {e_grad}')
                print(f'grad/d := {d_grad}')
                d_nonzero, d_params = nonzero_params(d_model)
                e_nonzero, e_params = nonzero_params(e_model)
                print(f'nonzero/ef := {e_nonzero/float(e_params)}')
                print(f'nonzero/df := {d_nonzero/float(d_params)}')
                print(f'nonzero/e := {e_nonzero}')
                print(f'nonzero/d := {d_nonzero}')
                with torch.no_grad():
                    val_samples = next(iter(val_dataloader))
                    val_features = get_features(val_samples).to(args.device)
                    val_entangled_features = get_features(val_samples, entangle=True).to(args.device)
                    val_labels = get_labels(val_samples, task).to(args.device)

                    e_out = copy_and_zero(e_model)(val_entangled_features)
                    d_out = copy_and_zero(d_model)(val_features)

                    e_pred = e_out > 0
                    e_acc = (e_pred == val_labels).float().mean()
                    d_pred = d_out > 0
                    d_acc = (d_pred == val_labels).float().mean()

                    print(f'val_acc/e := {e_acc}')
                    print(f'val_acc/d := {d_acc}')

                to_save = {
                    'd_model': d_model.state_dict(),
                    'e_model': e_model.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'e_opt': e_opt.state_dict()
                }
                torch.save(to_save, 'checkpoint.pt')

            step += 1
        print(f'lr/e := {e_sched.get_lr()[0]}')
        print(f'lr/d := {d_sched.get_lr()[0]}')
        e_sched.step()
        d_sched.step()

if __name__ == '__main__':
    run()
