# Whisper_LCM
Whisper to convert speech to text, then use text as prompt to LCM

```
conda create -n sd python=3.10
conda activate sd
conda install ffmpeg

pip install openvino-2023.2.0-12778-cp310-cp310-win_amd64.whl
pip install torch==2.1.0 torchvision
pip install diffusers transformers omegaconf accelerate
pip install numpy opencv-python pillow
pip install gradio==3.41.1 
pip install sentencepiece
pip install SpeechRecognition
```

## 运行代码：
`python audiosd_v0.1.py`
