from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import yaml
import numpy as np
import time
import pathlib
import os


# get config file
with open('config.yml', 'r') as file:
    inference_config = yaml.safe_load(file)

# base model
if "sd_xl_base" in inference_config['models'][0]:
    sd_pipeline = StableDiffusionXLPipeline
else:
    sd_pipeline = StableDiffusionPipeline
pipe = sd_pipeline.from_single_file("./{}".format(inference_config['models'][0])).to("cuda")

# dataset name
dataset_name = inference_config["datasets"][0]

# prompts
prompts = inference_config["prompts"]

# model checkpoints
epoch_min = inference_config["loop_epochs"]["epoch_min"]
epoch_max = inference_config["loop_epochs"]["epoch_max"]

for prompt in prompts:
    output_path = "outputs/ds={}_prompt={}_seed={}".format(dataset_name, prompt.replace(" ", "-"), inference_config['seed'])
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    for k in range(epoch_min, epoch_max):
        number = '{}'.format(k)
        number_filled = number.zfill(6)
        pipe.load_lora_weights(
            "./Loras/{}/output/{}-{}.safetensors".format(dataset_name, dataset_name, number_filled),
            weight_name="{}-{}.safetensors".format(dataset_name, number_filled)) 

        scl_min = inference_config["loop_scales"]["scale_min"]
        scl_max = inference_config["loop_scales"]["scale_max"]
        step = inference_config["loop_scales"]["resolution"]
        lora_scales = np.arange(scl_min, scl_max, (scl_max-scl_min)/step)
        for lscale in lora_scales:

            image = pipe(
                prompt, 
                num_inference_steps=30, 
                cross_attention_kwargs={"scale": lscale}, 
                generator=torch.manual_seed(inference_config['seed']),
            ).images[0]

            # timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(
                output_path, "image_ds={}_prompt={}_model={}_scale={:.4f}.jpg".format(
                    dataset_name, 
                    prompt.replace(" ", "-"),
                    number_filled, 
                    lscale))
            print("saving:", filename)
            image.save(filename)