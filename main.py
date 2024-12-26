import os
from swapper import process, load_models
from restoration import *
import numpy as np
from PIL import Image
import cv2
import torch

def face_swap(source_img, target_img, output_path):
    model = "./checkpoints/inswapper_128.onnx"
    
    face_analyser, face_swapper = load_models(model)
    check_ckpts()
    upsampler = set_realesrgan()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                     codebook_size=1024,
                                                     n_head=8,
                                                     n_layers=9,
                                                     connect_list=["32", "64", "128", "256"],
                                                    ).to(device)
    ckpt_path = "checkpoints/CodeFormer/codeformer.pth"
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    try:
        target_img = Image.open(target_img)
        source_img = Image.open(source_img)
        result_image = process(source_img, target_img, -1, -1, face_swapper, face_analyser)
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, 
                                        False, 
                                        True, 
                                        1, 
                                        0.5,
                                        upsampler,
                                        codeformer_net,
                                        device)
        result_image = Image.fromarray(result_image)
        result_image.save(output_path)
        print(f'Result saved successfully: {output_path}')
    
    except AttributeError:
        print(f"Can Not Process {target_img}")

def main():
    source_img = "test0.jpg"
    target_img = "test1.jpg"
    output_path = "Output_image.jpg"

    face_swap(source_img, target_img, output_path)

if __name__ == '__main__':
    main()
