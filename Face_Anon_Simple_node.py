# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import torch
import numpy as np
from diffusers import AutoencoderKL, DDPMScheduler
from .src.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from .src.referencenet.unet_2d_condition import UNet2DConditionModel
from .src.pipelines.pipeline_referencenet import StableDiffusionReferenceNetPipeline

from .node_utils import download_weights,load_images_list,pil2narry,tensor_to_pil,tensor_upscale
from comfy.model_management import unload_all_models
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))


# make face anon folder
weigths_face_anon_current_path = os.path.join(folder_paths.models_dir, "Face_anon_simple")
if not os.path.exists(weigths_face_anon_current_path):
    os.makedirs(weigths_face_anon_current_path)

face_anon_unet_path = os.path.join(weigths_face_anon_current_path, "unet")
if not os.path.exists(face_anon_unet_path):
    os.makedirs(face_anon_unet_path)

face_anon_refer_path = os.path.join(weigths_face_anon_current_path, "referencenet")
if not os.path.exists(face_anon_refer_path):
    os.makedirs(face_anon_refer_path)

face_anon_cond_path = os.path.join(weigths_face_anon_current_path, "conditioning_referencenet")
if not os.path.exists(face_anon_cond_path):
    os.makedirs(face_anon_cond_path)

try:
    folder_paths.add_model_folder_path("Face_anon_simple", weigths_face_anon_current_path, False)
except:
    folder_paths.add_model_folder_path("Face_anon_simple", weigths_face_anon_current_path)


class Face_Anon_Simple_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "repo": ("STRING", {"default": "hkung/face-anon-simple"}),
                "vae":(["none"] + folder_paths.get_filename_list("vae"),),
                "lowvram": ("BOOLEAN", {"default": True},),
            }
        }

    RETURN_TYPES = ("FACEANON_PIPE",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Face_Anon_Simple"

    def main_loader(self,repo,vae,lowvram):
        # base model
        if repo=="hkung/face-anon-simple":
            unet = UNet2DConditionModel.from_pretrained(
                repo, subfolder="unet", use_safetensors=True
            )
            referencenet = ReferenceNetModel.from_pretrained(
                repo, subfolder="referencenet", use_safetensors=True
            )
            conditioning_referencenet = ReferenceNetModel.from_pretrained(
                repo, subfolder="conditioning_referencenet", use_safetensors=True
            )
            print("***use repo,load checkpoints successfully ***")
        else:
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="unet", use_safetensors=True
                )
                referencenet = ReferenceNetModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="referencenet", use_safetensors=True
                )
                conditioning_referencenet = ReferenceNetModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="conditioning_referencenet", use_safetensors=True
                )
                print(f"***use local {weigths_face_anon_current_path},load checkpoints successfully ***")
            except:
                print("***miss checkpoints ,auto dowload from hub***")
                download_weights(weigths_face_anon_current_path,"hkung/face-anon-simple",subfolder="unet",pt_name="diffusion_pytorch_model.safetensors")
                download_weights(weigths_face_anon_current_path, "hkung/face-anon-simple", subfolder="unet",
                                 pt_name="config.json")
                download_weights(weigths_face_anon_current_path, "hkung/face-anon-simple", subfolder="conditioning_referencenet",
                                 pt_name="diffusion_pytorch_model.safetensors")
                download_weights(weigths_face_anon_current_path, "hkung/face-anon-simple", subfolder="conditioning_referencenet",
                                 pt_name="config.json")
                download_weights(weigths_face_anon_current_path, "hkung/face-anon-simple", subfolder="referencenet",
                                 pt_name="diffusion_pytorch_model.safetensors")
                download_weights(weigths_face_anon_current_path, "hkung/face-anon-simple", subfolder="referencenet",
                                 pt_name="config.json")
               
                unet = UNet2DConditionModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="unet", use_safetensors=True
                )
                referencenet = ReferenceNetModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="referencenet", use_safetensors=True
                )
                conditioning_referencenet = ReferenceNetModel.from_pretrained(
                    weigths_face_anon_current_path, subfolder="conditioning_referencenet", use_safetensors=True
                )
                print(f"***download {weigths_face_anon_current_path},load checkpoints successfully ***")
        

        # pre vae
        if vae=="none":
            raise "**** need choice a vae checkpoinst! ****"
        vae_id = folder_paths.get_full_path("vae", vae)
        vae_config = os.path.join(current_path, "sd_repo", "vae")
        VAE = AutoencoderKL.from_single_file(vae_id, config=vae_config, use_safetensors=True)
        print("***load vae successfully ***")
       
        scheduler = DDPMScheduler.from_pretrained(os.path.join(current_path, "sd_repo"), subfolder="scheduler", use_safetensors=True)
        
        pipe = StableDiffusionReferenceNetPipeline(
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            vae=VAE,
            scheduler=scheduler,
        )
       
        if lowvram:
            pipe.enable_model_cpu_offload()
        model={"pipe":pipe,"lowvram":lowvram}
        return (model,)


class Face_Anon_Simple_Align:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "width": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64, "display": "number"}),
                "align": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "cond_image": ("IMAGE",),  # [B,H,W,C], C=3
            }
        }
    
    RETURN_TYPES = ("CONDITIONING","FACEANON_ALIGN",)
    RETURN_NAMES = ("condition","face_align",)
    FUNCTION = "main"
    CATEGORY = "Face_Anon_Simple"
    
    def main(self,clip_vision,image,width,height, align,**kwargs):
        D, _, _, _ = image.size()
        if D > 1:
            original_image = list(torch.chunk(image, chunks=D))[0]
        else:
            original_image = tensor_upscale(image,width,height)
        
        cond=clip_vision.encode_image(original_image)["image_embeds"]
        original_image=tensor_to_pil(original_image)
        cond_image = kwargs.get("cond_image")
        
        Swap_faces=False
        
        if isinstance(cond_image,torch.Tensor):
            E, _, _, _ = cond_image.size()
            if E > 1:
                cond_image = list(torch.chunk(cond_image, chunks=E))[0]
            else:
                cond_image = tensor_upscale(cond_image, width, height)
            cond_cond=clip_vision.encode_image(cond_image)["image_embeds"]
            cond_image = tensor_to_pil(cond_image)
            Swap_faces=True
        else:
            cond_cond=cond
            cond_image=original_image
            
        condition={"cond":cond,"cond_cond":cond_cond,"original_image":original_image,"cond_image":cond_image,"Swap_faces":Swap_faces}
        unload_all_models()
        if align:
            import face_alignment
            face_align = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, face_detector="sfd"
            )
        else:
            face_align=None
        return (condition,face_align)

class Face_Anon_Simple_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FACEANON_PIPE",),
                "condition": ("CONDITIONING",),  # [B,H,W,C], C=3
                "seed": ("INT", {"default": 0, "min": 0, "max":MAX_SEED}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "anonymization_degree": ("FLOAT", {"default": 1.25, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
            "optional":{
                "face_align": ("FACEANON_ALIGN",),
                        }
           
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "Face_Anon_Simple"
    
    def main(self, model,condition,seed,cfg, steps,anonymization_degree,**kwargs):

        conditioning_image=condition.get("cond_image")
        original_image=condition.get("original_image")
        image_emb = condition.get("cond")
        cond_imag_emb = condition.get("cond_cond")
        Swap_faces= condition.get("Swap_faces")
        
        pipe=model.get("pipe")
        lowvram=model.get("lowvram")
        
        if not lowvram:
            pipe.to(device)
            
        face_align = kwargs.get("face_align")
        
        if face_align:
            from .node_utils  import anonymize_faces_in_image
            images = anonymize_faces_in_image(
                image=original_image,
                face_alignment=face_align,
                pipe=pipe,
                generator=torch.manual_seed(seed),
                face_image_size=512,
                num_inference_steps=steps,
                guidance_scale=cfg,
                anonymization_degree=anonymization_degree,
                image_emb=image_emb,
                cond_imag_emb=cond_imag_emb if Swap_faces else image_emb,
            )
        else:
            images = pipe(
                source_image=original_image,
                conditioning_image=conditioning_image if Swap_faces else original_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.manual_seed(seed),
                anonymization_degree=anonymization_degree,
                image_emb=image_emb,
                cond_imag_emb=cond_imag_emb if Swap_faces else image_emb,
            ).images[0]

        if not lowvram:
            pipe.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        images=load_images_list(images) if isinstance(images,list) else pil2narry(images)
        
        return (images,)


NODE_CLASS_MAPPINGS = {
    "Face_Anon_Simple_LoadModel":Face_Anon_Simple_LoadModel,
    "Face_Anon_Simple_Align":Face_Anon_Simple_Align,
    "Face_Anon_Simple_Sampler": Face_Anon_Simple_Sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Face_Anon_Simple_LoadModel":"Face_Anon_Simple_LoadModel",
    "Face_Anon_Simple_Align":"Face_Anon_Simple_Align",
    "Face_Anon_Simple_Sampler": "Face_Anon_Simple_Sampler",
}
