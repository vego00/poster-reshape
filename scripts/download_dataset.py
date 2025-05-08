import os
import json
import argparse
from datasets import load_dataset
from PIL import Image

def download_pokemon_captions(dataset_name: str, split: str, out_dir: str):
    # 출력 디렉토리 생성
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 데이터셋 로드
    ds = load_dataset(dataset_name, split=split)

    # 매핑을 저장할 리스트
    mapping = []

    # 순회하며 이미지 저장 및 메타 수집
    for idx, item in enumerate(ds):
        img = item["image"]  # Image 객체
        caption = item["text"]

        # 파일명: 0000.png, 0001.png, …
        fname = f"{idx:04d}.png"
        path = os.path.join(images_dir, fname)
        img.save(path)

        mapping.append({
            "image": os.path.relpath(path),
            "caption": caption
        })

    # JSON으로 매핑 저장
    meta_path = os.path.join(out_dir, "captions.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(mapping)}장 이미지와 캡션을 '{out_dir}'에 저장했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset to local files")
    parser.add_argument(
        "--dataset",
        type=str,
        default="diffusers/pokemon-gpt4-captions",
        help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/pokemon",
        help="Output directory for images + captions.json"
    )
    args = parser.parse_args()

    download_pokemon_captions(
        dataset_name=args.dataset,
        split=args.split,
        out_dir=args.out_dir
    )
