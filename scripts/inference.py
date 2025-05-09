import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTokenizer
from tqdm import tqdm

# 설정
base_model_id = "runwayml/stable-diffusion-v1-5"
lora_weights_path = "./models/lora-checkpoints"
input_dir = "./data/vertical"
output_dir = "./outputs/generated"
prompt_template = "wide cinematic poster, movie: {title}, {description}"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(output_dir, exist_ok=True)

# 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# LoRA 가중치 로드
pipe.unet.load_attn_procs(
    lora_weights_path,
    use_safetensors=True,
    local_files_only=True
)

# 이미지 리스트 및 메타 정보
import json
with open("data/pairs.json", "r") as f:
    metadata = json.load(f)

# 추론 루프
for item in tqdm(metadata):
    input_path = os.path.join(input_dir, item["vertical"])
    image = Image.open(input_path).convert("RGB")

    # 프롬프트 생성
    prompt = prompt_template.format(title=item["title"], description=item["description"])

    # 이미지 생성
    result = pipe(prompt=prompt, height=512, width=896, num_inference_steps=50).images[0]

    # 저장
    save_path = os.path.join(output_dir, f"{item['title'].replace(' ', '_')}_wide.jpg")
    result.save(save_path)
