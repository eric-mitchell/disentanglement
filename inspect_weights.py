import torch


checkpoint = '/iris/u/em7/code/disentanglement/env/.guild/runs/292806ab4448488e97eeb62bccb67217/checkpoint.pt'


def run():
    d = torch.load(checkpoint)
    d_w0 = d['d_model']['seq.0.weight']
    d_w1 = d['d_model']['seq.2.weight']
    d_b0 = d['d_model']['seq.0.bias']
    d_b1 = d['d_model']['seq.2.bias']

    d_w0_ = d['d_model']['seq.0.weight'].clone()
    d_w1_ = d['d_model']['seq.2.weight'].clone()
    d_b0_ = d['d_model']['seq.0.bias'].clone()
    d_b1_ = d['d_model']['seq.2.bias'].clone()

    d_w0_[d_w0_.abs() < 5e-4] = 0
    d_w1_[d_w1_.abs() < 5e-4] = 0
    d_b0_[d_b0_.abs() < 5e-4] = 0
    d_b1_[d_b1_.abs() < 5e-4] = 0

    print('d_w0', d_w0[d_w0.abs() > 5e-4])
    print('d_w0', torch.where(d_w0.abs() > 5e-4))
    print('d_w1', d_w1[d_w1.abs() > 5e-4])
    print('d_w1', torch.where(d_w1.abs() > 5e-4))
    print('d_b0', d_b0[d_b0.abs() > 5e-4])
    print('d_b0', torch.where(d_b0.abs() > 5e-4))
    print('d_b1', d_b1[d_b1.abs() > 5e-4])
    print('d_b1', torch.where(d_b1.abs() > 5e-4))

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    run()
