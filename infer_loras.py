from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

# pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
# pipe = StableDiffusionXLPipeline.from_single_file("./sd_xl_base_1.0.safetensors").to("cuda")
pipe = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly.safetensors").to("cuda")

dataset_name = "dataset-chair"

for k in range(1, 7):
    for lora_weight_frac in range(1, 21):

        lora_weight = lora_weight_frac / 20.0 * 2.0

        number = '{}'.format(k)
        number_filled = number.zfill(6)
        print('load:', "{}-{}.safetensors".format(dataset_name, number_filled))
        pipe.load_lora_weights(
            "./Loras/{}/output/{}-{}.safetensors".format(dataset_name, dataset_name, number_filled),
            weight_name="{}-{}.safetensors".format(dataset_name, number_filled)) 

        prompt = "foam divorce"

        lora_scale = lora_weight
        seed = 2048

        image = pipe(
            prompt, 
            num_inference_steps=30, 
            cross_attention_kwargs={"scale": lora_scale}, 
            generator=torch.manual_seed(seed),
        ).images[0]

        image.save("image_model={}_scale={:.2f}.jpg".format(number_filled, lora_weight))