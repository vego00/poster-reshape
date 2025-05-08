import os
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
import time
from datetime import timedelta

# 설정
model_id = "runwayml/stable-diffusion-v1-5"
dataset_repo = "your-username/movie-poster-pairs"
output_dir = "./models/lora-checkpoints"
lora_rank = 4
max_train_steps = 1000
train_batch_size = 1
image_size = 512

# 가속기 초기화
accelerator = Accelerator()
device = accelerator.device

# 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe.to(device)

# 토크나이저, 텍스트 인코더
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet

# LoRA 설정
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=8,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    bias="none",
    task_type="UNET"
)
unet = get_peft_model(unet, lora_config)

# 데이터셋 로드
dataset = load_dataset(dataset_repo, split="train")

# 데이터 전처리 함수
def preprocess(example):
    image = Image.open(example["vertical"]).convert("RGB").resize((image_size, image_size))
    prompt = f"horizontal movie poster, {example['title']}, {example['description']}"
    example["pixel_values"] = pipe.feature_extractor(images=image, return_tensors="pt")["pixel_values"][0]
    example["prompt_ids"] = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"][0]
    return example

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 데이터로더
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    input_ids = torch.stack([x["prompt_ids"] for x in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# 학습 루프
unet.train()
start_time = time.time()
batch_times = []

for step, batch in enumerate(tqdm(train_dataloader, total=max_train_steps)):
    if step >= max_train_steps:
        break
        
    batch_start = time.time()
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)

    loss = pipe.unet(pixel_values, input_ids=input_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    batch_time = time.time() - batch_start
    batch_times.append(batch_time)
    
    if step % 100 == 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        remaining_steps = max_train_steps - (step + 1)
        estimated_time = avg_batch_time * remaining_steps
        print(f"Step {step} - Loss: {loss.item():.4f}")
        print(f"  평균 배치 처리 시간: {avg_batch_time:.2f}초")
        print(f"  예상 남은 시간: {timedelta(seconds=int(estimated_time))}")
        print(f"  예상 완료 시간: {timedelta(seconds=int(time.time() - start_time + estimated_time))}")

# LoRA 가중치 저장
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"LoRA weights saved to {output_dir}")
