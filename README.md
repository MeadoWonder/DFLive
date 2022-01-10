## DFLive
#### Quick Start
1. 安装docker镜像：选择mltooling/ml-workspace-gpu:0.13.2，配置文件中打开runtime: nvidia，设置映射端口
2. 拉取仓库：git clone https://github.com/MeadoWonder/DFLive.git --recursive
3. 安装依赖：pip install --upgrade opencv-python opencv-contrib-python onnx onnxruntime-gpu
4. 下载模型：在https://github.com/iperov/DeepFaceLive/releases下载dfm模型到server/models/文件夹
5. 运行：python server.py [--port your-server-port] [--model your-dfm-model-filename]  
example1: python server.py  
example2: python server.py --port 10001 --model Kim_Jarrey.dfm  
等待直到“Models loaded and ready to work...”