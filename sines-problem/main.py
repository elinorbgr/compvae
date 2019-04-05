#!/usr/bin/env python3
import data
import models
import torch
import math

ROOT = './out/'

DEVICE = 'cuda'
#DEVICE = 'cpu'

BATCHLEN = 256
BATCH_SPLIT = 4
ITERNUM = 500001

DUMP_PERIOD = 100
MODEL_DUMP_PERIOD = 10000

X_SIGMA = None

def main():
    if X_SIGMA is not None:
        vae = models.VAE(x_logvar=2*math.log(X_SIGMA)).to(device=DEVICE)
    else:
        vae = models.VAE().to(device=DEVICE)
    optimizer = torch.optim.Adamax(vae.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: max(0.5 ** e, 0.01))

    # run loop
    with open(ROOT+'out.csv', 'w') as logfile:
        print('batch;wi losses;z loss;x loss;ELBO;', file=logfile)
        for b in range(ITERNUM):
            if b%20000 == 0:
                scheduler.step()

            vae.zero_grad()
            min_curves = 1
            max_curves = max(2, min(16, 1+int(b/2500)))
            # average the loss over different batch sizes
            loss_w = torch.zeros((), device=DEVICE)
            loss_z = torch.zeros((), device=DEVICE)
            loss_x = torch.zeros((), device=DEVICE)
            for _ in range(BATCH_SPLIT):
                (fs, curves) = data.generate_batch(BATCHLEN//BATCH_SPLIT, freqrg=(1,10), nfreqrg=(min_curves, max_curves), device=DEVICE)
                (l_w, l_z, l_x, K) = vae.rec_losses(curves, fs)
                loss_w += l_w / BATCH_SPLIT
                loss_z += l_z / BATCH_SPLIT
                loss_x += l_x / BATCH_SPLIT
            elbo = loss_w + loss_z + loss_x
            print("{};{:.2e};{:.2e};{:.2e};{:.2e}".format(b, float(loss_w), float(loss_z), float(loss_x), float(elbo)), file=logfile)
            logfile.flush()
            elbo.backward()
            # update gradients
            optimizer.step()
            if b % DUMP_PERIOD == 0:
                (mus, sigmas) = vae.rec_output(curves, fs)
                with open(ROOT+'curve-{:06d}.csv'.format(b), 'w') as outfile:
                    print('x;mu;sigma', file=outfile)
                    for i in range(curves.size(1)):
                        print("{};{:.3f};{:.3f};{:.3f}".format(i, curves[0,i], mus[0,i], sigmas[0,i]), file=outfile)
                xs = vae.generate_same_latent(fs[0:1,:], 4)
                with open(ROOT+'gen-{:06d}.csv'.format(b), 'w') as outfile:
                    for i in range(xs.size(1)):
                        print("{}".format(i), file=outfile, end='')
                        for j in range(xs.size(0)):
                            print(";{:.3f}".format(xs[j,i]), file=outfile, end='')
                        print("", file=outfile)

            if b % MODEL_DUMP_PERIOD == 0:
                torch.save(vae.state_dict(), ROOT + 'dump-{:06d}.pt'.format(b))


if __name__ == "__main__":
    print("ROOT = {}".format(ROOT))
    main()
