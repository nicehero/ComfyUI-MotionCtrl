# MotionCtrl: A Unified and Flexible Motion Controller for Video Generation

[![ Paper](https://img.shields.io/badge/Paper-gray
)](https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf) &ensp; [![ arXiv](https://img.shields.io/badge/arXiv-red
)](https://arxiv.org/pdf/2312.03641.pdf) &ensp; [![Porject Page](https://img.shields.io/badge/Project%20Page-green
)
](https://wzhouxiff.github.io/projects/MotionCtrl/) &ensp; [![ Demo](https://img.shields.io/badge/Gradio%20Demo-orange
)](https://huggingface.co/spaces/TencentARC/MotionCtrl)

---

🔥🔥  This is an implementation of MotionCtrl for ComfyUI

Download the weights of MotionCtrl  [motionctrl.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) and put it to `ComfyUI/models/checkpoints`

One node "Motionctrl Sample"

unofficial implementation "MotionCtrl deployed on AnimateDiff" workflow:

https://github.com/chaojie/ComfyUI-MotionCtrl/blob/main/workflow_motionctrl.json

1. Generate LVDM/VideoCrafter Video
2. Images->Scribble
3. Use AnimateDiff Scribble SparseCtrl
