import torch
from utils.utils import compute_loss
from models import Darknet

targets = torch.load('targets.pt').float()
# model = torch.load('model.pt').cpu()
model = Darknet("cfg/yolov3-tiny-1cls-mobilenetv2.cfg")
x = torch.rand(1, 3, 416, 416)
p = model(x)

p = torch.load('p.pt')
for i in range(len(p)):
    p[i] = p[i].float()



loss, _ = compute_loss(p, targets, model)



