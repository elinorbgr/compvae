import torch
import math
from torchvision.utils import save_image

import multiprocessing as mp

import data
import models

ROOT = './out/'

DEVICE='cuda'
#DEVICE='cpu'

BATCHLEN = 256
BATCH_SPLIT = 4
ITERNUM = 500001
MODEL_DUMP_PERIOD = 1000

CONTINUE = None

def main():
    print("ROOT = {}".format(ROOT))
    vae = models.VAE().to(device=DEVICE)
    optimizer = torch.optim.Adamax(vae.parameters(), lr=2e-5, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: max(0.5 ** e, 0.05))
    if CONTINUE is not None:
        vae.load_state_dict(torch.load(ROOT + 'dump-{:06d}.pt'.format(CONTINUE)))
        # pre-advance the learning rate
        for _ in range(CONTINUE//20000):
            scheduler.step()
        start_iter = CONTINUE + 1
    else:
        start_iter = 0

    # run loop
    with open(ROOT+'out.csv', 'w') as logfile:
        print("batch;wi losses;z loss;x loss;ELBO", file=logfile)
        for b in range(start_iter, ITERNUM):
            if b%20000 == 0:
                scheduler.step()

            vae.zero_grad()
            bmin = 1
            bmax = max(2, min(6, 1+int(b/2500)))
            loss_w = torch.zeros((), device=DEVICE)
            loss_z = torch.zeros((), device=DEVICE)
            loss_x = torch.zeros((), device=DEVICE)
            images = []
            recs = []
            gens = []
            for _ in range(BATCH_SPLIT):
                (img, col_labels, loc_labels) = data.generate_batch(BATCHLEN//BATCH_SPLIT, (bmin, bmax), device=DEVICE)
                (rec_img, l_w, l_z, l_x, K) = vae.rec_with_losses(img, (col_labels, loc_labels))
                loss_w += l_w / BATCH_SPLIT
                loss_z += l_z / BATCH_SPLIT
                loss_x += l_x / BATCH_SPLIT
                images.append(img)
                recs.append(rec_img.detach())
                with torch.no_grad():
                    gens.append(vae.generate((col_labels, loc_labels)))
            elbo = loss_w+loss_z+loss_x
            print("{};{:.2e};{:.2e};{:.2e};{:.2e}".format(b, float(loss_w), float(loss_z), float(loss_x), float(elbo)), file=logfile)
            logfile.flush()
            # some flukes can make the loss get incredibly big in rare occasions, ignore them
            if math.isfinite(float(elbo)) and float(elbo) < 1e8:
                elbo.backward()
                optimizer.step()

            if b%100 == 0:
                save_image(torch.cat(images, dim=0), ROOT+"{0:06d}-orig.png".format(b), nrow=16)
                save_image(torch.cat(recs, dim=0), ROOT+"{0:06d}-rec.png".format(b), nrow=16)
                save_image(torch.cat(gens, dim=0), ROOT+"{0:06d}-gen.png".format(b), nrow=16)

            if b % MODEL_DUMP_PERIOD == 0:
                torch.save(vae.state_dict(), ROOT + 'dump-{:06d}.pt'.format(b))


if __name__ == "__main__":
    main()
