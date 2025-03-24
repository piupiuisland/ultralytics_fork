

from ultralytics import YOLO
import torch
from collections import OrderedDict
from ultralytics.nn.modules.head import Detect
import copy
import yaml

# Load a model
model_name = "Ax_Yolov8n_4267"
model_name = "Ax_Yolov8mLite_4990"
# model_name = "yolov8x"

# model = YOLO(f"{model_name}.pt")  # load a pretrained model (recommended for training)

trained_pt = "/home/ubuntu/work_root/ruhui_download_repo/ultralytics_newest_forYOLOv11/ultralytics_fork/runs/detect/train37/weights/last.pt"
model= YOLO(trained_pt)
print(model)

breakpoint()
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")



def _rewrite_head_output(model, num_cls, dataset_names):

    chs = [model.model.model[-1].cv3[i][2].in_channels for i in range(len(model.model.model[-1].cv3))]
    new_head = Detect(nc=num_cls, ch=chs)
    head_weights = copy.deepcopy(model.model.model[-1].state_dict())

    setattr(model.model.model[-1], "nc", num_cls)
    setattr(model.model.model[-1], "no", num_cls+4*model.model.model[-1].reg_max)
    setattr(model, "nc", num_cls)
    # dataset_names={0:"place"}
    setattr(model.model, "names", dataset_names)

    for i in range(3):
        in_ch = model.model.model[-1].cv3[i][2].in_channels
        model.model.model[-1].cv3[i][2] = torch.nn.Conv2d(in_ch, num_cls, kernel_size=(1, 1), stride=(1, 1))
    # model.model.model[-1].__init__(nc=num_cls, ch=chs)

    return model


# def _rewrite_head_output(model, num_cls):

#     chs = [model.model.model[-1].cv3[i][2].in_channels for i in range(len(model.model.model[-1].cv3))]
#     new_head = Detect(nc=num_cls, ch=chs)

#     # dynamic = False  # force grid reconstruction
#     # export = False  # export mode
#     # format = None  # export format
#     # end2end = False  # end2end
#     # max_det = 300  # max_det
#     # shape = None
#     # anchors = torch.empty(0)  # init
#     # strides = torch.empty(0)  # init
#     # legacy = False  # backward compatibility for v3/v5/v8/v9 models

#     setattr(new_head, "nc", model.model.model[-1].dynamic)
#     setattr(new_head, "strides", model.model.model[-1].strides)
#     setattr(new_head, "stride", model.model.model[-1].stride)
#     setattr(new_head, "anchors", model.model.model[-1].anchors)
#     setattr(new_head, "shape", model.model.model[-1].shape)
#     setattr(new_head, "legacy", model.model.model[-1].legacy)

#     new_head.load_state_dict(model.model.model[-1].state_dict(), strict=True)

#     model.model.model[-1] = new_head
#     return model


count_parameters(model.model)
print(model)
# # print("new modelw conv0: ", model.model.state_dict()["model.0.conv.weight"][0,1,:,:])


# data_yaml = "coco8.yaml"
# metrics = model.val(data=data_yaml, batch=16, rect=False)  # evaluate model performance on the validation set

# # path = model.export(format="onnx", opset=11, dynamic=False)  # export the model to ONNX format



## pose model results

"""
using env:

./host.qtools/venv-dev/bin/python3
/home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov5/bin/python

source /home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov5/bin/activate

#or

/home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov5_310_torch113/bin/python

source /home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov5_310_torch113/bin/activate


#or
/home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov9_py310_torch24/bin/python

source /home/ubuntu/work_root/ruhui_download_repo/my_envs/yolov9_py310_torch24/bin/activate

# or

/home/ubuntu/work_root/ruhui_download_repo/ruhui_env/py310_torch24/bin/python

source /home/ubuntu/work_root/ruhui_download_repo/ruhui_env/py310_torch24/bin/activate
"""