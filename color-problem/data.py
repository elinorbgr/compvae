import torch
import math
import random

COLORS = torch.tensor([
        [1.0, 0.0, 0.0], # RED
        [0.0, 1.0, 0.0], # GREEN
        [0.0, 0.0, 1.0], # BLUE
        [0.0, 0.0, 0.0], # BLACK
        [1.0, 1.0, 1.0], # WHITE
    ]
)

def draw_shade(locations, colors):
    batchlen = locations.size(0)
    K = locations.size(1)
    xs = locations[:,:,0:1]
    ys = locations[:,:,1:2]
    line = torch.arange(32, device=locations.device).float()*2.0/31.0 - 1.0 # from -1 to +1
    line = line.view(1,1,32)
    x_distance = ((line-xs)**2).view(batchlen,K,1,32)
    y_distance = ((line-ys)**2).view(batchlen,K,32,1)
    distance_grid = x_distance + y_distance
    intensity_factor = distance_grid.new_zeros((batchlen, K, 1, 1)).uniform_(5, 10)
    #intensity_factor = 5.0
    intensity = torch.softmax(- intensity_factor * distance_grid, dim=1)
    return torch.sum(intensity.view(batchlen, K, 1, 32, 32) * colors.view(batchlen, K, 3, 1, 1), dim=1)

def generate_batch(batchlen, blobcount, device):
    (bmin, bmax) = blobcount
    num_blobs = random.randrange(bmin, bmax+1)
    img = torch.ones((batchlen, 3, 32, 32), device=device)
    labels = torch.randint(COLORS.shape[0], size=(batchlen, num_blobs), device=device)
    locations = torch.rand(batchlen, num_blobs, 2, device=device) * 1.8 - 0.9
    img = draw_shade(locations, COLORS.to(device=device)[labels, :])
    return (torch.clamp(img ** (1/2.4), 0.0, 1.0), labels, locations)
