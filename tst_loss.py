import torch
from utils.utils import compute_loss


targets = torch.load('targets.pt').float()
model = torch.load('model.pt')
p = torch.load('p.pt')
for i in range(len(p)):
    p[i] = p[i].float()



loss, _ = compute_loss(p, targets, model)



