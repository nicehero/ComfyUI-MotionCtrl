import argparse
import datetime
import glob
import json
import math
import os
import tempfile
import folder_paths
import imageio
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
## note: decord should be imported after torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from .lvdm.models.samplers.ddim import DDIMSampler
from .main.evaluation.motionctrl_prompts_camerapose_trajs import (
    both_prompt_camerapose_traj, cmcm_prompt_camerapose, omom_prompt_traj)
from .main.evaluation.motionctrl_inference import motionctrl_sample,save_images,load_camera_pose,load_trajs,load_model_checkpoint,post_prompt,DEFAULT_NEGATIVE_PROMPT
from .utils.utils import instantiate_from_config
from .gradio_utils.traj_utils import process_points,get_flow

def process_camera(camera_pose_str,frame_length):
    RT=json.loads(camera_pose_str)
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
    
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT

    
def process_traj(points_str,frame_length):
    points=json.loads(points_str)
    for i in range(frame_length):
        if len(points)<=i:
            points.append(points[len(points)-1])
    xy_range = 1024
    #points = process_points(points,frame_length)
    points = [[int(256*x/xy_range), int(256*y/xy_range)] for x,y in points]
    
    optical_flow = get_flow(points,frame_length)
    # optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow
    
def save_results(video, fps=10):
    
    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [t, h, w*n, 3]
    
    path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    outframes=[]
    
    #writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps)
    for i in range(grid.shape[0]):
        img = grid[i].numpy()
        image_tensor_out = torch.tensor(np.array(grid[i]).astype(np.float32) / 255.0)  # Convert back to CxHxW
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        outframes.append(image_tensor_out)
        #writer.append_data(img)

    #writer.close()
    return torch.cat(tuple(outframes), dim=0).unsqueeze(0)

MOTION_CAMERA_OPTIONS = ["U", "D", "L", "R", "O", "O_0.2x", "O_0.4x", "O_1.0x", "O_2.0x", "O_0.2x", "O_0.2x", "Round-RI", "Round-RI_90", "Round-RI-120", "Round-ZoomIn", "SPIN-ACW-60", "SPIN-CW-60", "I", "I_0.2x", "I_0.4x", "I_1.0x", "I_2.0x", "1424acd0007d40b5", "d971457c81bca597", "018f7907401f2fef", "088b93f15ca8745d", "b133a504fc90a2d1"]

MOTION_TRAJ_OPTIONS = ["curve_1", "curve_2", "curve_3", "curve_4", "horizon_2", "shake_1", "shake_2", "shaking_10"]

        
def read_points(file, video_len=16, reverse=False):
    with open(file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(',')
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]
    
    return points
    
class LoadMotionCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_camera": (MOTION_CAMERA_OPTIONS,),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_camera_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_camera_preset(self, motion_camera):
        data="[]"
        with open(f'custom_nodes/ComfyUI-MotionCtrl/examples/camera_poses/test_camera_{motion_camera}.json') as f:
            data = f.read()
        return (data,)
        

class LoadMotionTrajPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_traj": (MOTION_TRAJ_OPTIONS,),
                "frame_length": ("INT", {"default": 16}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_traj_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_traj_preset(self, motion_traj, frame_length):
        points = read_points(f'custom_nodes/ComfyUI-MotionCtrl/examples/trajectories/{motion_traj}.txt',frame_length)
        return (json.dumps(points),)

class MotionctrlSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "prompt": ("STRING", {"multiline": True, "default":"a rose swaying in the wind"}),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2],[1,0,0,0,0,1,0,0,0,0,1,0.28750000000000003],[1,0,0,0,0,1,0,0,0,0,1,0.37500000000000006],[1,0,0,0,0,1,0,0,0,0,1,0.4625000000000001],[1,0,0,0,0,1,0,0,0,0,1,0.55],[1,0,0,0,0,1,0,0,0,0,1,0.6375000000000002],[1,0,0,0,0,1,0,0,0,0,1,0.7250000000000001],[1,0,0,0,0,1,0,0,0,0,1,0.8125000000000002],[1,0,0,0,0,1,0,0,0,0,1,0.9000000000000001],[1,0,0,0,0,1,0,0,0,0,1,0.9875000000000003],[1,0,0,0,0,1,0,0,0,0,1,1.0750000000000002],[1,0,0,0,0,1,0,0,0,0,1,1.1625000000000003],[1,0,0,0,0,1,0,0,0,0,1,1.2500000000000002],[1,0,0,0,0,1,0,0,0,0,1,1.3375000000000001],[1,0,0,0,0,1,0,0,0,0,1,1.4250000000000003],[1,0,0,0,0,1,0,0,0,0,1,1.5125000000000004]]"}),
                "traj": ("STRING", {"multiline": True, "default":"[[117, 102],[114, 102],[109, 102],[106, 102],[105, 102],[102, 102],[99, 102],[97, 102],[96, 102],[95, 102],[93, 102],[89, 102],[85, 103],[82, 103],[81, 103],[80, 103],[79, 103],[78, 103],[76, 103],[74, 104],[73, 104],[72, 104],[71, 104],[70, 105],[69, 105],[68, 105],[67, 105],[66, 106],[64, 107],[63, 108],[62, 108],[61, 108],[61, 109],[60, 109],[59, 109],[58, 109],[57, 110],[56, 110],[55, 111],[54, 111],[53, 111],[52, 111],[52, 112],[51, 112],[50, 112],[50, 113],[49, 113],[48, 113],[46, 114],[46, 115],[45, 115],[45, 116],[44, 116],[43, 117],[42, 117],[41, 117],[41, 118],[40, 118],[41, 118],[41, 119],[42, 119],[43, 119],[44, 119],[46, 119],[47, 119],[48, 119],[49, 119],[50, 119],[51, 119],[52, 119],[53, 119],[54, 119],[55, 119],[56, 118],[58, 118],[59, 118],[61, 118],[63, 118],[64, 117],[67, 117],[70, 117],[71, 117],[73, 117],[75, 116],[76, 116],[77, 116],[80, 116],[82, 116],[83, 116],[84, 116],[85, 116],[88, 116],[91, 116],[94, 116],[97, 116],[98, 116],[100, 116],[101, 117],[102, 117],[104, 117],[105, 117],[106, 117],[107, 117],[108, 117],[109, 117],[110, 117],[111, 117],[115, 117],[119, 117],[123, 117],[124, 117],[128, 117],[129, 117],[132, 117],[134, 117],[135, 117],[136, 117],[138, 117],[139, 117],[140, 117],[141, 117],[142, 116],[145, 116],[146, 116],[148, 116],[149, 116],[151, 115],[152, 115],[153, 115],[154, 115],[155, 114],[156, 114],[157, 114],[158, 114],[159, 114],[162, 114],[163, 113],[164, 113],[165, 113],[166, 113],[167, 113],[168, 113],[169, 113],[170, 113],[171, 113],[172, 113],[173, 113],[174, 113],[175, 113],[178, 113],[181, 113],[182, 113],[183, 113],[184, 113],[185, 113],[187, 113],[188, 113],[189, 113],[191, 113],[192, 113],[193, 113],[194, 113],[195, 113],[196, 113],[197, 113],[198, 113],[199, 113],[200, 113],[201, 113],[202, 113],[203, 113],[202, 113],[201, 113],[200, 113],[198, 113],[197, 113],[196, 113],[195, 112],[194, 112],[193, 112],[192, 112],[191, 111],[190, 111],[189, 111],[188, 110],[187, 110],[186, 110],[185, 110],[184, 110],[183, 110],[182, 110],[181, 110],[180, 110],[179, 110],[178, 110],[177, 110],[175, 110],[173, 110],[172, 110],[171, 110],[170, 110],[168, 110],[167, 110],[165, 110],[164, 110],[163, 110],[161, 111],[159, 111],[155, 111],[153, 111],[151, 111],[151, 112],[150, 112],[149, 112],[148, 112],[147, 112],[145, 112],[143, 113],[142, 113],[140, 113],[139, 113],[138, 113],[136, 113],[135, 113],[134, 113],[133, 114],[131, 114],[130, 114],[128, 115],[127, 115],[126, 115],[125, 115],[124, 115],[122, 115],[121, 115],[120, 115],[118, 116],[115, 116],[113, 116],[111, 116],[109, 117],[106, 117],[103, 117],[102, 117],[100, 117],[98, 117],[97, 117],[95, 117],[94, 117],[93, 117],[92, 117],[91, 117],[90, 117],[89, 117],[88, 117],[87, 117],[86, 117],[85, 117],[84, 117],[83, 117],[84, 117],[85, 117],[87, 117],[88, 117],[89, 117],[90, 117],[92, 117],[93, 117],[95, 117],[97, 117],[99, 117],[101, 117],[103, 117],[104, 117],[105, 117],[106, 117],[107, 117],[108, 117],[109, 117],[110, 117],[112, 117],[113, 117],[114, 117],[116, 117],[117, 117],[118, 117],[119, 117],[120, 117],[121, 117],[123, 117],[124, 117],[125, 117],[126, 117],[127, 117],[129, 117],[130, 117],[131, 117],[133, 117],[134, 117],[135, 117],[136, 117],[137, 117],[138, 117],[139, 117],[140, 117],[141, 117],[142, 117],[143, 117],[145, 117],[146, 117],[147, 117],[148, 117],[149, 117],[150, 117],[149, 117],[148, 117],[147, 117],[146, 117],[144, 117],[143, 118],[142, 118],[141, 118],[140, 118],[139, 118],[138, 118],[136, 118],[135, 118],[132, 119],[131, 119],[130, 119],[129, 119],[127, 119],[126, 119],[124, 119],[123, 119],[122, 119],[121, 119],[119, 119],[118, 119],[117, 119],[115, 119],[114, 119],[113, 119],[112, 119],[111, 119],[110, 119],[109, 119],[108, 119],[107, 119],[106, 119],[107, 119],[108, 119],[109, 119],[110, 119],[112, 119],[113, 119],[114, 119],[115, 119],[116, 119],[117, 119],[118, 119],[119, 119],[120, 119],[121, 119],[122, 119],[123, 119],[124, 119],[125, 119],[126, 119],[127, 119],[127, 119],[127, 119],[127, 119]]"}),
                "frame_length": ("INT", {"default": 16}),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 1234}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"
        
    def run_inference(self,ckpt_name,prompt,camera,traj,frame_length,steps,seed):
        gpu_num=1
        gpu_no=0
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        comfy_path = os.path.dirname(folder_paths.__file__)
        config_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl/configs/inference/config_both.yaml')

        args={"savedir":f'./output/both_seed20230211',"ckpt_path":f"{ckpt_path}","adapter_ckpt":None,"base": f"{config_path}","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":1234,"cond_T":800,"save_imgs":True,"cond_dir":"./custom_nodes/ComfyUI-MotionCtrl/examples/"}

        prompts = prompt
        RT = process_camera(camera,frame_length).reshape(-1,12)
        traj_flow = process_traj(traj,frame_length).transpose(3,0,1,2)
        print(prompts)
        print(RT.shape)
        print(traj_flow.shape)
        
        args["savedir"]=f'./output/{args["condtype"]}_seed{args["seed"]}'
        config = OmegaConf.load(args["base"])
        OmegaConf.update(config, "model.params.unet_config.params.temporal_length", frame_length)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(gpu_no)
        assert os.path.exists(args["ckpt_path"]), f'Error: checkpoint {args["ckpt_path"]} Not Found!'
        print(f'Loading checkpoint from {args["ckpt_path"]}')
        model = load_model_checkpoint(model, args["ckpt_path"], args["adapter_ckpt"])
        model.eval()
       
        ## run over data
        assert (args["height"] % 16 == 0) and (args["width"] % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        
        ## latent noise shape
        h, w = args["height"] // 8, args["width"] // 8
        channels = model.channels
        frames = model.temporal_length
        #frames = frame_length
        noise_shape = [args["bs"], channels, frames, h, w]

        savedir = os.path.join(args["savedir"], "samples")
        os.makedirs(savedir, exist_ok=True)
        
        #noise_shape = [1, 4, 16, 32, 32]
        unconditional_guidance_scale = 7.5
        unconditional_guidance_scale_temporal = None
        n_samples = 1
        ddim_steps= steps
        ddim_eta=1.0
        cond_T=800
        #seed = args["seed"]

        if n_samples < 1:
            n_samples = 1
        if n_samples > 4:
            n_samples = 4

        seed_everything(seed)
        
        camera_poses = RT
        trajs = traj_flow
        camera_poses = torch.tensor(camera_poses).float()
        trajs = torch.tensor(trajs).float()
        camera_poses = camera_poses.unsqueeze(0)
        trajs = trajs.unsqueeze(0)
        if torch.cuda.is_available():
            camera_poses = camera_poses.cuda()
            trajs = trajs.cuda()
        
        ddim_sampler = DDIMSampler(model)
        batch_size = noise_shape[0]
        prompts=prompt
        ## get condition embeddings (support single prompt only)
        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(len(prompts)):
            prompts[i] = f'{prompts[i]}, {post_prompt}'

        cond = model.get_learned_conditioning(prompts)
        if camera_poses is not None:
            RT = camera_poses[..., None]
        else:
            RT = None

        traj_features = None
        if trajs is not None:
            traj_features = model.get_traj_features(trajs)
        else:
            traj_features = None
            
        uc = None
        if unconditional_guidance_scale != 1.0:
            # prompts = batch_size * [""]
            prompts = batch_size * [DEFAULT_NEGATIVE_PROMPT]
            uc = model.get_learned_conditioning(prompts)
            if traj_features is not None:
                un_motion = model.get_traj_features(torch.zeros_like(trajs))
            else:
                un_motion = None
            uc = {"features_adapter": un_motion, "uc": uc}
        else:
            uc = None
        
        batch_images=[]
        batch_variants = []
        for _ in range(n_samples):
            if ddim_sampler is not None:
                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                                features_adapter=traj_features,
                                                pose_emb=RT,
                                                cond_T=cond_T
                                                )        
            #print(f'{samples}')
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_variants = batch_variants[0]
        
        ret = save_results(batch_variants, fps=10)
        #print(ret)
        return ret
        
        
class ImageSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "selected_indexes": ("STRING", {
                    "multiline": False,
                    "default": "1,2,3"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    OUTPUT_NODE = False

    CATEGORY = "motionctrl"

    def run(self, images: torch.Tensor, selected_indexes: str):
        shape = images.shape
        len_first_dim = shape[0]

        selected_index: list[int] = []
        total_indexes: list[int] = list(range(len_first_dim))
        for s in selected_indexes.strip().split(','):
            try:
                if ":" in s:
                    _li = s.strip().split(':', maxsplit=1)
                    _start = _li[0]
                    _end = _li[1]
                    if _start and _end:
                        selected_index.extend(
                            total_indexes[int(_start):int(_end)]
                        )
                    elif _start:
                        selected_index.extend(
                            total_indexes[int(_start):]
                        )
                    elif _end:
                        selected_index.extend(
                            total_indexes[:int(_end)]
                        )
                else:
                    x: int = int(s.strip())
                    if x < len_first_dim:
                        selected_index.append(x)
            except:
                pass

        if selected_index:
            print(f"ImageSelector: selected: {len(selected_index)} images")
            return (images[selected_index, :, :, :], )

        print(f"ImageSelector: selected no images, passthrough")
        return (images, )


NODE_CLASS_MAPPINGS = {
    "Motionctrl Sample":MotionctrlSample,
    "Load Motion Camera Preset":LoadMotionCameraPreset,
    "Load Motion Traj Preset":LoadMotionTrajPreset,
    "Select Image Indices": ImageSelector
}