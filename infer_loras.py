from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

# pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
# pipe = StableDiffusionXLPipeline.from_single_file("./sd_xl_base_1.0.safetensors").to("cuda")
pipe = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly.safetensors").to("cuda")

for k in range(1, 11):

    number = '{}'.format(k)
    number_filled = number.zfill(6)
    print('load:', "lauras-{}.safetensors".format(number_filled))
    pipe.load_lora_weights(
        "./Loras/lauras/output/lauras-{}.safetensors".format(number_filled),
        weight_name="lauras-{}.safetensors".format(number_filled)) 

    prompt = "ceramic divorce"

    lora_scale = 1.2
    seed = 2048

    image = pipe(
        prompt, 
        num_inference_steps=30, 
        cross_attention_kwargs={"scale": lora_scale}, 
        generator=torch.manual_seed(seed),
    ).images[0]

    image.save("image-model={}.jpg".format(number_filled))