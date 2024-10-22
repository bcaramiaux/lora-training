from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import yaml
import numpy as np

with open('config.yml', 'r') as file:
    inference_config = yaml.safe_load(file)


if "sd_xl_base" in inference_config['model']:
    sd_pipeline = StableDiffusionXLPipeline
else:
    sd_pipeline = StableDiffusionPipeline
pipe = sd_pipeline.from_single_file("./{}".format(inference_config['model'])).to("cuda")

dataset_name = inference_config["dataset"]

for k in range(inference_config["loop_epochs"]["epoch_min"],
               inference_config["loop_epochs"]["epoch_max"]):

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

        prompt = "foam divorce"

        image = pipe(
            prompt, 
            num_inference_steps=30, 
            cross_attention_kwargs={"scale": lscale}, 
            generator=torch.manual_seed(2048),
        ).images[0]

        image.save("image_model={}_scale={:.2f}.jpg".format(number_filled, lora_weight))