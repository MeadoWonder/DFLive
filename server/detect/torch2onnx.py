import torch


x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
torch_model = torch.load('./data/resnet50-finetuned.pth').eval()
torch_model(x)

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./data/detect_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],     # the model's input names
                  output_names=['output']   # the model's output names
                  )
