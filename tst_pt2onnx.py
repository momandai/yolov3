import torch
import torchvision
import torch.onnx
from models import Darknet

model_name = 'yolov3-tiny-1cls-mobilenetv2.onnx'

# model = torch.load("weights/yolov3-tiny-1cls-mobilenetv2.pt")
trained_dict = torch.load("weights/yolov3.pt")['model']
# model = Darknet("cfg/yolov3-tiny-1cls-mobilenetv2.cfg")
model = Darknet("cfg/yolov3.cfg")
model.load_state_dict(trained_dict)
# model.eval()
dummy_input = torch.rand(1, 3, 416, 416)
torch_out = torch.onnx._export(model, dummy_input, model_name, export_params=True)

