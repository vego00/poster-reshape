import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file

def get_unique_filename(base_path):
    """동일한 파일명이 있을 경우 넘버링을 추가하여 고유한 파일명을 반환합니다."""
    if not os.path.exists(base_path):
        return base_path
        
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while os.path.exists(os.path.join(directory, f"{name}_{counter}{ext}")):
        counter += 1
        
    return os.path.join(directory, f"{name}_{counter}{ext}")

def main():
    # 설정
    base_model_id = "runwayml/stable-diffusion-v1-5"
    lora_weights_path = os.path.abspath("./models/lora-checkpoints")
    input_dir = "./data/vertical"
    output_dir = "./outputs/generated"
    prompt_template = "wide cinematic poster, movie: {title}, {description}"

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파이프라인 로드
    print("모델 로딩 중...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    # LoRA 설정
    print("LoRA 설정 중...")
    lora_config = LoraConfig(
        r=4,  # 학습 시와 동일한 rank
        lora_alpha=4,
        target_modules=["to_v", "to_out.0", "to_q", "to_k"],  # adapter_config.json과 동일한 순서
        bias="none"
    )
    
    # UNet에 LoRA 적용
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights_path)
    
    # LoRA 가중치 확인
    print("\nLoRA 가중치 확인:")
    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            print(f"LoRA 파라미터 {name}:")
            print(f"- 평균값: {param.data.mean().item():.6f}")
            print(f"- 표준편차: {param.data.std().item():.6f}")
            print(f"- 최소값: {param.data.min().item():.6f}")
            print(f"- 최대값: {param.data.max().item():.6f}")
            print("---")
    
    # 데이터셋 준비
    print("\n데이터셋 준비 중...")
    metadata = []
    pairs_dir = "./data/pairs"
    
    # data/pairs 디렉토리에서 모든 영화 폴더를 순회
    for movie_name in tqdm(os.listdir(pairs_dir), desc="영화 폴더 로딩"):
        movie_dir = os.path.join(pairs_dir, movie_name)
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
                metadata.append({
                    'vertical': vertical_path,
                    'horizontal': horizontal_path,
                    'title': movie_name,
                    'description': f"movie poster for {movie_name}"
                })
    
    print(f"데이터셋 준비 완료: {len(metadata)}개 항목")

    # 추론 루프
    print("\n이미지 생성 시작...")
    for item in tqdm(metadata, desc="이미지 생성"):
        try:
            # 입력 이미지 로드
            input_path = item["vertical"]
            if not os.path.exists(input_path):
                print(f"입력 이미지를 찾을 수 없음: {input_path}")
                continue

            image = Image.open(input_path).convert("RGB")

            # 프롬프트 생성
            prompt = prompt_template.format(
                title=item["title"],
                description=item["description"]
            )

            # 이미지 생성
            result = pipe(
                prompt=prompt,
                height=512,
                width=896,
                num_inference_steps=50
            ).images[0]

            # 저장 경로 생성 및 저장
            base_save_path = os.path.join(
                output_dir,
                f"{item['title'].replace(' ', '_')}_wide.jpg"
            )
            save_path = get_unique_filename(base_save_path)
            result.save(save_path)
            print(f"생성된 이미지 저장: {save_path}")

        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {e}")
            continue

    print("모든 이미지 생성 완료!")

if __name__ == "__main__":
    main() 