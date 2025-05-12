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
    def __init__(self, data_dir, tokenizer, image_size=512, max_pairs=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"데이터셋 로딩 시작: {data_dir}")
        
        # data/pairs 디렉토리에서 모든 영화 폴더를 순회
        for movie_name in tqdm(os.listdir(data_dir), desc="영화 폴더 로딩"):
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
                    
                    # max_pairs가 설정되어 있고, 해당 개수에 도달하면 중단
                    if max_pairs and len(self.pairs) >= max_pairs:
                        break
            if max_pairs and len(self.pairs) >= max_pairs:
                break

        print(f"데이터셋 로딩 완료: {len(self.pairs)}개 페어")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 이미지 로드 및 전처리
        with Image.open(pair['vertical']) as vertical_img:
            vertical_img = vertical_img.convert('RGB').resize((self.image_size, self.image_size))
            pixel_values = self.transform(vertical_img)
            
        with Image.open(pair['horizontal']) as horizontal_img:
            horizontal_img = horizontal_img.convert('RGB').resize((self.image_size, self.image_size))
            target_pixel_values = self.transform(horizontal_img)
        
        # 채널 수를 4로 확장 (RGB + alpha)
        pixel_values = torch.cat([pixel_values, torch.ones_like(pixel_values[:1])], dim=0)
        target_pixel_values = torch.cat([target_pixel_values, torch.ones_like(target_pixel_values[:1])], dim=0)
        
        # 프롬프트 생성 및 토큰화
        prompt = f"horizontal movie poster, {pair['movie']}"
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"][0]
        
        return {
            "pixel_values": pixel_values,
            "prompt_ids": prompt_ids,
            "target_pixel_values": target_pixel_values
        }

def main():
    # CUDA 메모리 설정
    if torch.cuda.is_available():
        # 메모리 단편화 방지
        torch.cuda.set_per_process_memory_fraction(0.6)  # 60%로 제한
        # 메모리 할당자 설정
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # 캐시 비우기
        torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='Train LoRA for Poster Reshape')
    parser.add_argument('--data_dir', type=str, default='data/pairs',
                      help='Directory containing the paired poster dataset')
    parser.add_argument('--output_dir', type=str, default='./models/lora-checkpoints',
                      help='Directory to save LoRA checkpoints')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
                      help='Base model ID')
    parser.add_argument('--lora_rank', type=int, default=2,  # 랭크 감소
                      help='LoRA rank')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size (주의: GPU 메모리 사용량에 따라 조절 필요)')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Image size for training')
    parser.add_argument('--max_pairs', type=int, default=4,
                      help='최대 로드할 페어 수 (메모리 제한 시 사용)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='데이터로더 워커 수')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.6,
                      help='GPU 메모리 사용 비율 (0.0 ~ 1.0)')
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
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    
    # 메모리 최적화를 위한 설정
    if torch.cuda.is_available():
        # VAE를 CPU로 이동
        pipe.vae.to("cpu")
        # 텍스트 인코더를 GPU로 이동
        pipe.text_encoder.to(device)
        # UNet만 GPU에 유지
        pipe.unet.to(device)
        
        # 메모리 사용량 출력
        print("\n초기 메모리 사용량:")
        print(f"- UNet: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
    print("모델 로딩 완료")

    # 토크나이저, 텍스트 인코더
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # LoRA 설정
    print("LoRA 설정 중...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=4,  # alpha 값 감소
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    print("LoRA 설정 완료")

    # 모델을 float16으로 설정
    unet = unet.to(dtype=torch.float16)
    text_encoder = text_encoder.to(dtype=torch.float16)

    # 데이터셋 및 데이터로더 생성
    print("데이터셋 생성 중...")
    dataset = PosterPairDataset(args.data_dir, tokenizer, args.image_size, args.max_pairs)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"배치 크기: {args.batch_size}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # 학습 루프
    print("학습 시작...")
    unet.train()
    start_time = time.time()
    last_print_time = start_time
    print_interval = 100
    
    # 학습 시작 전 메모리 사용량 출력
    if torch.cuda.is_available():
        print("\n학습 시작 전 메모리 사용량:")
        print(f"- 할당된 메모리: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"- 캐시된 메모리: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
        print(f"- 배치 크기 {args.batch_size}로 학습을 시작합니다.")
        print("  메모리 부족 시 배치 크기를 1로 낮추세요.")

    # 전체 예상 시간 계산
    total_samples = len(dataset)
    total_batches = len(train_dataloader) * args.num_epochs
    print(f"\n학습 정보:")
    print(f"- 전체 데이터 수: {total_samples}")
    print(f"- 배치 크기: {args.batch_size}")
    print(f"- 총 배치 수: {total_batches}")
    print(f"- 총 에폭 수: {args.num_epochs}")

    # AMP 스케일러 설정
    scaler = torch.amp.GradScaler()

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(tqdm(train_dataloader)):
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["prompt_ids"].to(device)
            target_pixel_values = batch["target_pixel_values"].to(device, dtype=torch.float16)
            
            # 텍스트 임베딩 생성
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                text_embeddings = text_encoder(input_ids)[0]
            
            # 노이즈 생성
            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device).long()
            
            # 노이즈가 추가된 이미지
            noisy_images = pipe.scheduler.add_noise(pixel_values, noise, timesteps)
            
            # UNet forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                model_pred = unet(noisy_images, timesteps, text_embeddings).sample
            
            # 손실 계산
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            
            # AMP를 사용한 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 100 스텝마다 진행 상황 출력
            if step % print_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                steps_per_second = (step + 1) / elapsed_time
                remaining_steps = total_batches - (step + 1)
                estimated_remaining_time = remaining_steps / steps_per_second
                
                print(f"\nStep {step}/{total_batches} - Loss: {loss.item():.4f}")
                print(f"  경과 시간: {timedelta(seconds=int(elapsed_time))}")
                print(f"  스텝/초: {steps_per_second:.2f}")
                print(f"  예상 남은 시간: {timedelta(seconds=int(estimated_remaining_time))}")
                print(f"  예상 완료 시간: {timedelta(seconds=int(current_time + estimated_remaining_time))}")
                print(f"  진행률: {(step + 1) / total_batches * 100:.1f}%")
                
                # 메모리 사용량 출력
                if torch.cuda.is_available():
                    print(f"  GPU 메모리: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                    torch.cuda.empty_cache()  # 메모리 정리

    # LoRA 가중치 저장
    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(args.output_dir)
    print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()
