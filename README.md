ComfyUI_Face_Anon_Simple:Face Anonymization make simple and easy.
---
* Origin from [face_anon_simple](https://github.com/hanweikung/face_anon_simple)   
  喜欢这个项目的，请给Face_Anon_Simple项目一个星星！（If you like this project, please give the Face_Anon_Simple project a star!）   

----

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Face_Anon_Simple.git
```
2.requirements  
----
```
pip install -r requirements.txt
```

3.checkpoints 
----
3.1 Face_Anon_Simple checkpoint ([huggingface](https://huggingface.co/hkung/face-anon-simple/tree/main))     
* keep repo fill in "hkung/face-anon-simple" will auto download from hb..   
默认为hkung/face-anon-simple时会自动下载模型，一般是C盘。 
* or 或者     
* keep repo empty,will auto download from hb to below folder:  
删掉repo的内容，保持空白，会自动下载到以下路径，换而言之，你提前下好放进去也可以。  
```
├── ComfyUI/models/Face_anon_simple
|   ├── unet
|      ├── config.json  
|      ├── diffusion_pytorch_model.safetensors  #3.2G
|   ├── referencenet
|      ├── config.json  
|      ├── diffusion_pytorch_model.safetensors  #3.2G
|   ├── conditioning_referencenet
|      ├── config.json  
|      ├── diffusion_pytorch_model.safetensors  #3.2G
```

3.2 vae and clip  
* 常规的 sd1.5  vae 和 clip_vision   
```
├── ComfyUI/models/vae
|     ├── vae-ft-mse-840000-ema-pruned.safetensors # or another sd 1.5 or sd21 vae
├── ComfyUI/models/clip_vision
|     ├──clip_vision_g.safetensors     #base openai/clip-vit-large-patch14
```
3.3
* when turn on align ,will download "face_alignment" checkpoints (2DFAN4-cd938726ad.zip ，91.8M)  
  开启面部对齐，会自动下载face_alignment的模型文件（2DFAN4-cd938726ad.zip），大小91.8M  


4 Function
--
* Anonymize images with a single aligned face / 微调单张脸  
  only link img to image，keep align turn off，changge degree / 单独连一张图，保持align 关闭，修改degree  
* Anonymize images with one or multiple unaligned faces / 微调单图多张脸   
   only link img to image，keep align turn on，changge degree / 单独连一张多脸图，保持align 开启，修改degree  
* Swap faces between two images / 借鉴它图微调面部   
   link condition image and image，keep align turn off，changge degree / 连俩个图片入口，输出是cond为底子，保持align关闭，修改degree   

5 Example
---
change face and keep  lighting and tone .  
![](https://github.com/smthemex/ComfyUI_Face_Anon_Simple/blob/main/exampleA.png)

6.Citation
------
**Face_anon_simple**
[link](https://github.com/hanweikung/face_anon_simple)


