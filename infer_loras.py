from diffusers import DiffusionPipeline
import torch

# pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe = DiffusionPipeline.from_single_file("./sd_xl_base_1.0.safetensors").to("cuda")

pipe.load_lora_weights(weight_name="./Loras/lauras/output/lauras-000002.safetensors") 

prompt = "ceramic divorce"

lora_scale = 1.2
image = pipe(
    prompt, 
    num_inference_steps=30, 
    cross_attention_kwargs={"scale": lora_scale}, 
    generator=torch.manual_seed(0)
).images[0]

