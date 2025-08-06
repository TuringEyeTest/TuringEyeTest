"""
NOTE: modify from https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/master/demo_smooth_grad_threshold.py
"""
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from einops import rearrange
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class GradCAM():
    def __init__(self, model, target_layer, input_token_len, output_ids):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.input_token_len = input_token_len
        self.output_ids = output_ids
        
        self.target_ids = self.output_ids[0][self.input_token_len:]
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        """保存特征图"""
        self.feature_maps = output
        output.retain_grad()
        num_token = self.feature_maps.shape[1]
        h = int(np.sqrt(num_token))


    def save_gradients(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
        # print(self.gradients)

    def generate_cam(self, image, inputs, image_len, text_len):
        self.model.eval()
        self.model.zero_grad()
        out = self.model(**inputs) 

        logits_shape = out.logits.shape
        target_logits = torch.sum(out.logits[0, self.input_token_len:self.output_ids.shape[1],:][torch.arange(len(self.target_ids)),self.target_ids.int()])

        target_logits.retain_grad()
        target_logits.backward(retain_graph=True)

        print(self.feature_maps.shape)

        num_token = self.feature_maps.shape[1]
        h = int(np.sqrt(image_len))

        self.feature_maps = rearrange(self.feature_maps.detach(),'(h w) c -> c h w ',w=h * 2,h=h * 2)
        self.gradients = rearrange(self.gradients.detach(),'(h w) c -> h w c',w=h * 2,h=h * 2)
        self.gradients = nn.ReLU()(self.gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
        activation = self.feature_maps

        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation.to(dtype=torch.float32), dim=0).squeeze().cpu().numpy().astype(np.float32)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        threshold = 0.5
        heatmap[heatmap < threshold] = 0
        
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_image = image
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return heatmap, superimposed_img


class SmoothGradCAM(GradCAM):
    def __init__(self, model, target_layer, input_token_len, output_ids, num_samples=50, noise_std=0.1):
        super().__init__(model, target_layer, input_token_len, output_ids)
        self.num_samples = num_samples
        self.noise_std = noise_std 

    def add_noise(self, tensor, noise_std):
        """向输入张量添加高斯噪声"""
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise

    def generate_smooth_cam(self, image, inputs, image_len, text_len):
        base_cam, _ = self.generate_cam(image, inputs=inputs, image_len=image_len, text_len=text_len) 
        
        smooth_cam = np.zeros_like(base_cam)
        for _ in range(self.num_samples):
            noisy_image = self.add_noise(image.clone(), self.noise_std)
            noisy_cam, _ = self.generate_cam(noisy_image, inputs, image_len, text_len)
            smooth_cam += noisy_cam

        smooth_cam /= self.num_samples
        smooth_cam = np.maximum(smooth_cam, 0)
        smooth_cam /= np.max(smooth_cam)
        smooth_cam = cv2.resize(smooth_cam, (image.size(3), image.size(2)))
        smooth_cam = np.uint8(255 * smooth_cam)
        smooth_cam = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET)

        original_image = image
        superimposed_img = smooth_cam * 0.4 + original_image * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return smooth_cam, superimposed_img

def setup_seeds():
    seed = 520

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

setup_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


task = "ColorBlind-2bit-hard"
image_id = '9'
image_path = f"yourpath/{task}/images/{image_id}.jpg"
save_path = f'{task}/{image_id}_vision_smooth_threshold_qwenvl7b_192'
image = Image.open(image_path).resize((192,192), resample=Image.Resampling.BOX)

# ========================================
#             Model Initialization
# ========================================
model_path = "yourpath/Qwen2.5-VL-7B-Instruct"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

for name, param in model.named_parameters():
    param.requires_grad_(True)
    print(name, param.requires_grad)
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are given an image. Please identify the characters in this image. Your final answer should be enclosed in \\box{} (e.g., \\box{NE27A})."},
            {
                "type": "image",
                "image": image,
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(text)
image_inputs, video_inputs = process_vision_info(messages)

image_len = image_inputs[0].size[0] * image_inputs[0].size[1] // (28*28)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

ptext = processor(
    text=[text.split("<|image_pad|>")[0]],
    padding=True,
    return_tensors="pt",
)

text_len = ptext['input_ids'].shape[1]

print(f"image_len: {image_len}, text_len: {text_len}")



assert len(inputs.input_ids) == 1, inputs
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
input_token_len = [len(in_ids) for in_ids in inputs.input_ids][0]

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)

inputs_out = processor(
    text=[text + output_text[0]],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs_out = inputs_out.to("cuda")


for i in range(0,len(model.visual.blocks)):   
    gradcam = SmoothGradCAM(model, model.visual.blocks[i].norm2, input_token_len, generated_ids)
    heatmap, result = gradcam.generate_cam(image, inputs=inputs_out, image_len=image_len, text_len=text_len)
    path_cam_img=os.path.join(save_path,f"{image_path.split('/')[-2]}_layer_{i+1}.jpg")
    path_raw_img=os.path.join(save_path,f"{image_path.split('/')[-2]}.jpg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(path_cam_img,result)
