import argparse
import datetime
import glob
import json
import math
import os
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
from .main.evaluation.motionctrl_inference import motionctrl_sample,save_images,save_results,load_camera_pose,load_trajs,load_model_checkpoint
from .utils.utils import instantiate_from_config

class MotionctrlSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"
        
    def run_inference(self):
        gpu_num=1
        gpu_no=0
        args={"savedir":f'./outputs/both_seed20230211',"ckpt_path":"./models/checkpoints/motionctrl.pth","adapter_ckpt":None,"base":"./custom_nodes/ComfyUI-MotionCtrl/configs/inference/config_both.yaml","condtype":"both","prompt_dir":None,"n_samples":1,"ddim_steps":50,"ddim_eta":1.0,"bs":1,"height":256,"width":256,"unconditional_guidance_scale":1.0,"unconditional_guidance_scale_temporal":None,"seed":20230211,"cond_T":800,"save_imgs":True,"cond_dir":"./custom_nodes/ComfyUI-MotionCtrl/examples/"}
        args["savedir"]=f'./outputs/{args["condtype"]}_seed{args["seed"]}'
        config = OmegaConf.load(args["base"])
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
        noise_shape = [args["bs"], channels, frames, h, w]

        savedir = os.path.join(args["savedir"], "samples")
        os.makedirs(savedir, exist_ok=True)

        if args["condtype"] == 'camera_motion':
            prompt_list = cmcm_prompt_camerapose['prompts']
            camera_pose_list, pose_name = load_camera_pose(args["cond_dir"], cmcm_prompt_camerapose['camera_poses'])
            traj_list = None
            save_name_list = []
            for i in range(len(pose_name)):
                save_name_list.append(f"{pose_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
        elif args["condtype"] == 'object_motion':
            prompt_list = omom_prompt_traj['prompts']
            traj_list, traj_name = load_trajs(args["cond_dir"], omom_prompt_traj['trajs'])
            camera_pose_list = None
            save_name_list = []
            for i in range(len(traj_name)):
                save_name_list.append(f"{traj_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
        elif args["condtype"] == 'both':
            prompt_list = both_prompt_camerapose_traj['prompts']
            camera_pose_list, pose_name = load_camera_pose(args["cond_dir"], both_prompt_camerapose_traj['camera_poses'])
            traj_list, traj_name = load_trajs(args["cond_dir"], both_prompt_camerapose_traj['trajs'])
            save_name_list = []
            for i in range(len(pose_name)):
                save_name_list.append(f"{pose_name[i]}__{traj_name[i]}__{prompt_list[i].replace(' ', '_').replace(',', '')}")
        
        num_samples = len(prompt_list)
        samples_split = num_samples // gpu_num
        print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
        #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
        indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
        prompt_list_rank = [prompt_list[i] for i in indices]
        camera_pose_list_rank = None if camera_pose_list is None else [camera_pose_list[i] for i in indices]
        traj_list_rank = None if traj_list is None else [traj_list[i] for i in indices]
        save_name_list_rank = [save_name_list[i] for i in indices]
        
        start = time.time() 
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args["bs"])), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args["bs"]]
            camera_poses = None if camera_pose_list_rank is None else camera_pose_list_rank[indice:indice+args["bs"]]
            trajs = None if traj_list_rank is None else traj_list_rank[indice:indice+args["bs"]]
            save_name = save_name_list_rank[indice:indice+args["bs"]]
            print(f'prompts:{prompts},camera_poses:{camera_poses},trajs:{trajs}')
            print(f'Processing {save_name}')

            if camera_poses is not None:
                camera_poses = torch.stack(camera_poses, dim=0).to("cuda")
            if trajs is not None:
                trajs = torch.stack(trajs, dim=0).to("cuda")

            batch_samples = motionctrl_sample(
                model, 
                prompts, 
                noise_shape,
                camera_poses=camera_poses,
                trajs=trajs,
                n_samples=args["n_samples"],
                unconditional_guidance_scale=args["unconditional_guidance_scale"],
                unconditional_guidance_scale_temporal=args["unconditional_guidance_scale_temporal"],
                ddim_steps=args["ddim_steps"],
                ddim_eta=args["ddim_eta"],
                cond_T = args["cond_T"],
            )
            
            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                name = save_name[nn]
                if len(name) > 90:
                    name = name[:90]
                filename = f'{name}_{idx*args["bs"]+nn:04d}_randk{gpu_no}'
                
                save_results(samples, filename, savedir, fps=10)
                if args["save_imgs"]:
                    parts = save_name[nn].split('__')
                    if len(parts) == 2:
                        cond_name = parts[0]
                        prname = prompts[nn].replace(' ', '_').replace(',', '')
                        cur_outdir = os.path.join(savedir, cond_name, prname)
                    elif len(parts) == 3:
                        poname, trajname, _ = save_name[nn].split('__')
                        prname = prompts[nn].replace(' ', '_').replace(',', '')
                        cur_outdir = os.path.join(savedir, poname, trajname, prname)
                    else:
                        raise NotImplementedError
                    os.makedirs(cur_outdir, exist_ok=True)
                    save_images(samples, cur_outdir)
                if nn % 100 == 0:
                    print(f'Finish {nn}/{len(batch_samples)}')

        print(f'Saved in {args["savedir"]}. Time used: {(time.time() - start):.2f} seconds')

        return savedir


NODE_CLASS_MAPPINGS = {
    "Motionctrl Sample":MotionctrlSample
}