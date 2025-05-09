import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import json
from tqdm import tqdm

def main():
    # 설정
    base_model_id = "runwayml/stable-diffusion-v1-5"
    lora_weights_path = "./models/lora-checkpoints"
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

    # LoRA 가중치 로드
    print("LoRA 가중치 로딩 중...")
    try:
        pipe.unet.load_attn_procs(
            lora_weights_path,
            use_safetensors=True,
            local_files_only=True
        )
        print("LoRA 가중치 로드 완료")
    except Exception as e:
        print(f"LoRA 가중치 로드 실패: {e}")
        print(f"경로 {lora_weights_path}에 LoRA 가중치 파일이 있는지 확인해주세요.")
        return

    # 메타데이터 로드
    print("메타데이터 로딩 중...")
    try:
        with open("data/pairs.json", "r") as f:
            metadata = json.load(f)
        print(f"메타데이터 로드 완료: {len(metadata)}개 항목")
    except Exception as e:
        print(f"메타데이터 로드 실패: {e}")
        return

    # 추론 루프
    print("이미지 생성 시작...")
    for item in tqdm(metadata, desc="이미지 생성"):
        try:
            # 입력 이미지 로드
            input_path = os.path.join(input_dir, item["vertical"])
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

            # 저장
            save_path = os.path.join(
                output_dir,
                f"{item['title'].replace(' ', '_')}_wide.jpg"
            )
            result.save(save_path)
            print(f"생성된 이미지 저장: {save_path}")

        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {e}")
            continue

    print("모든 이미지 생성 완료!")

if __name__ == "__main__":
    main() 