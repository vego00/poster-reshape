import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
import argparse
import time
from datetime import timedelta
from torchvision import transforms

class PosterPairDataset(Dataset):
    def __init__(self, data_dir, tokenizer, image_size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.pairs = []
        
        print(f"데이터셋 로딩 시작: {data_dir}")
        
        # data/pairs 디렉토리에서 모든 영화 폴더를 순회
        for movie_name in os.listdir(data_dir):
            movie_dir = os.path.join(data_dir, movie_name)
            if not os.path.isdir(movie_dir):
                continue
                
            # 각 영화 폴더의 pair_* 폴더들을 순회
            for pair_dir in os.listdir(movie_dir):
                if not pair_dir.startswith('pair_'):
                    continue
                    
                pair_path = os.path.join(movie_dir, pair_dir)
                vertical_path = os.path.join(pair_path, 'vertical.jpg')
                horizontal_path = os.path.join(pair_path, 'horizontal.jpg')
                
                if os.path.exists(vertical_path) and os.path.exists(horizontal_path):
                    self.pairs.append({
                        'vertical': vertical_path,
                        'horizontal': horizontal_path,
                        'movie': movie_name
                    })
                    print(f"  페어 추가: {movie_name}/{pair_dir}")

        print(f"데이터셋 로딩 완료: {len(self.pairs)}개 페어")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 이미지 로드 및 전처리
        vertical_img = Image.open(pair['vertical']).convert('RGB').resize((self.image_size, self.image_size))
        horizontal_img = Image.open(pair['horizontal']).convert('RGB').resize((self.image_size, self.image_size))
        
        # 프롬프트 생성
        prompt = f"horizontal movie poster, {pair['movie']}"
        
        # 이미지를 텐서로 변환
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        pixel_values = transform(vertical_img)
        target_pixel_values = transform(horizontal_img)
        
        # 채널 수를 4로 확장 (RGB + alpha)
        pixel_values = torch.cat([pixel_values, torch.ones_like(pixel_values[:1])], dim=0)
        target_pixel_values = torch.cat([target_pixel_values, torch.ones_like(target_pixel_values[:1])], dim=0)
        
        # 프롬프트 토큰화
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"][0]
        
        return {
            "pixel_values": pixel_values,
            "prompt_ids": prompt_ids,
            "target_pixel_values": target_pixel_values
        }

def main():
    parser = argparse.ArgumentParser(description='Train LoRA for Poster Reshape')
    parser.add_argument('--data_dir', type=str, default='data/pairs',
                      help='Directory containing the paired poster dataset')
    parser.add_argument('--output_dir', type=str, default='./models/lora-checkpoints',
                      help='Directory to save LoRA checkpoints')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
                      help='Base model ID')
    parser.add_argument('--lora_rank', type=int, default=4,
                      help='LoRA rank')
    parser.add_argument('--max_train_steps', type=int, default=1000,
                      help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--image_size', type=int, default=512,
                      help='Image size for training')
    args = parser.parse_args()
    
    print("초기화 시작...")
    
    # 가속기 초기화
    accelerator = Accelerator()
    device = accelerator.device
    print(f"사용 장치: {device}")

    # 모델 로드
    print("모델 로딩 중...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(device)
    print("모델 로딩 완료")

    # 토크나이저, 텍스트 인코더
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # LoRA 설정
    print("LoRA 설정 중...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=8,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    print("LoRA 설정 완료")

    # 데이터셋 및 데이터로더 생성
    print("데이터셋 생성 중...")
    dataset = PosterPairDataset(args.data_dir, tokenizer, args.image_size)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"배치 크기: {args.batch_size}")

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # 학습 루프
    print("학습 시작...")
    unet.train()
    start_time = time.time()
    batch_times = []

    for step, batch in enumerate(tqdm(train_dataloader, total=args.max_train_steps)):
        if step >= args.max_train_steps:
            break
            
        batch_start = time.time()
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["prompt_ids"].to(device)
        target_pixel_values = batch["target_pixel_values"].to(device)
        
        # 텍스트 임베딩 생성
        text_embeddings = text_encoder(input_ids)[0]
        
        # 노이즈 생성
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device).long()
        
        # 노이즈가 추가된 이미지
        noisy_images = pipe.scheduler.add_noise(pixel_values, noise, timesteps)
        
        # UNet forward
        model_pred = unet(noisy_images, timesteps, text_embeddings).sample
        
        # 손실 계산
        loss = torch.nn.functional.mse_loss(model_pred, noise)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if step % 100 == 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_steps = args.max_train_steps - (step + 1)
            estimated_time = avg_batch_time * remaining_steps
            print(f"Step {step} - Loss: {loss.item():.4f}")
            print(f"  평균 배치 처리 시간: {avg_batch_time:.2f}초")
            print(f"  예상 남은 시간: {timedelta(seconds=int(estimated_time))}")
            print(f"  예상 완료 시간: {timedelta(seconds=int(time.time() - start_time + estimated_time))}")

    # LoRA 가중치 저장
    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(args.output_dir)
    print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()
