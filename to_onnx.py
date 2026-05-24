import torch
import torchvision.transforms as T
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import json
import sys

from mona.nn.model2 import Model2
# from mona.text import word_to_index, index_to_word
from mona.datagen import OnnxDataGen
from mona.config import config
from mona.text.construct_lexicon import get_lexicon

if len(sys.argv) == 1:
    print("Please input a model path")
    exit(0)

lexicon = get_lexicon(config["model_type"])
lexicon_size = lexicon.lexicon_size()

# name = "model_training.pt"
# name = "model_acc100-epoch15.pt"
name = sys.argv[1]
onnx_name = name.rsplit(".", 2)[0] + ".onnx"
# net = Model(len(word_to_index))
net = Model2(lexicon_size, in_channels=1)

net.load_state_dict(torch.load(f"models/{name}", weights_only=True))
net.eval()

to_tensor = T.ToTensor()
image_width = config["train_width"]
image_height = config["height"]
x = torch.zeros((1, 1, image_height, image_width))
# x = to_tensor(x)
print(x.shape)
# x.unsqueeze_(0) # (1, 3, 32, width)
y = net(x)      # (width / 8, 1, lexicon_size)
print(x.size(), y.size())

# onnx_program = torch.onnx.dynamo_export(net, x)
# onnx_program.save(f"models/{onnx_name}")

torch.onnx.export(
    net,
    x,
    f"models/{onnx_name}",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

onnx_model = onnx.load(f"models/{onnx_name}")
onnx.checker.check_model(onnx_model)

# onnxruntime 1.17.0 only supports IR version 9, but PyTorch exports IR version 10
# downgrade IR version for compatibility
onnx_model.ir_version = 9
onnx.save(onnx_model, f"models/{onnx_name}")

with open("models/index_2_word.json", "w", encoding="utf-8") as f:
    j = json.dumps(lexicon.index_to_word, indent=4, ensure_ascii=False).encode("utf8")
    f.write(j.decode())
