import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import argparse

# 설정
VERTICAL_DIR = "data/vertical"
HORIZONTAL_DIR = "data/horizontal"
OUTPUT_DIR = "data/pairs"
VISUALIZATION_DIR = "data/visualization"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_pairs(pairs, movie_name, show=False):
    """매칭된 이미지 쌍을 시각화합니다."""
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    # 시각화 결과 저장 디렉토리 생성
    vis_dir = os.path.join(VISUALIZATION_DIR, movie_name)
    os.makedirs(vis_dir, exist_ok=True)

    # 각 쌍마다 시각화
    for i, pair in enumerate(pairs):
        # 이미지 로드
        vertical_img = Image.open(pair["vertical"])
        horizontal_img = Image.open(pair["horizontal"])

        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(vertical_img)
        ax1.set_title(f"Vertical\nScore: {pair['score']:.4f}")
        ax1.axis('off')
        
        ax2.imshow(horizontal_img)
        ax2.set_title("Horizontal")
        ax2.axis('off')

        plt.suptitle(f"Pair {i+1}/{n_pairs} - {movie_name}")
        plt.tight_layout()

        # 결과 저장
        output_path = os.path.join(vis_dir, f"pair_{i+1}.png")
        plt.savefig(output_path)
        
        if show:
            plt.show()
        else:
            plt.close()

def copy_paired_images(pairs, movie_name):
    """매칭된 이미지 쌍을 새로운 폴더에 복사합니다."""
    if not pairs:
        return

    # 영화별 결과 디렉토리 생성
    movie_pairs_dir = os.path.join(OUTPUT_DIR, movie_name)
    os.makedirs(movie_pairs_dir, exist_ok=True)

    # 각 쌍마다 폴더 생성 및 이미지 복사
    for i, pair in enumerate(pairs):
        pair_dir = os.path.join(movie_pairs_dir, f"pair_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)

        # 이미지 복사
        vertical_dst = os.path.join(pair_dir, "vertical.jpg")
        horizontal_dst = os.path.join(pair_dir, "horizontal.jpg")
        
        shutil.copy2(pair["vertical"], vertical_dst)
        shutil.copy2(pair["horizontal"], horizontal_dst)

        # 점수 정보 저장
        with open(os.path.join(pair_dir, "score.txt"), "w") as f:
            f.write(f"Similarity Score: {pair['score']:.4f}")

    print(f"📁 매칭된 이미지 쌍이 `{movie_pairs_dir}`에 저장되었습니다.")

def process_movie_pair(movie_name, show_visualization=False, save_pairs=False):
    # 1. CLIP 모델 로드
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    # 2. 이미지 파일 로딩 및 임베딩 계산
    def encode_image(path):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        return feats.cpu().numpy().squeeze()

    # 가로/세로 리스트
    vertical_path = os.path.join(VERTICAL_DIR, movie_name)
    horizontal_path = os.path.join(HORIZONTAL_DIR, movie_name)
    
    if not os.path.exists(vertical_path) or not os.path.exists(horizontal_path):
        print(f"⚠️ Skipping {movie_name}: Missing vertical or horizontal directory")
        return
        
    vertical_files = sorted(os.listdir(vertical_path))
    horizontal_files = sorted(os.listdir(horizontal_path))

    if not vertical_files or not horizontal_files:
        print(f"⚠️ Skipping {movie_name}: No images found")
        return

    print(f"\nProcessing movie: {movie_name}")
    print(f"Vertical images: {len(vertical_files)}")
    print(f"Horizontal images: {len(horizontal_files)}")

    # 임베딩 사전
    vert_embs = []
    for fn in tqdm(vertical_files, desc="Encoding vertical"):
        vert_embs.append(encode_image(os.path.join(vertical_path, fn)))
    horz_embs = []
    for fn in tqdm(horizontal_files, desc="Encoding horizontal"):
        horz_embs.append(encode_image(os.path.join(horizontal_path, fn)))

    V = np.stack(vert_embs)       # shape (N, 512)
    H = np.stack(horz_embs)       # shape (M, 512)

    # 3. 유사도 행렬 (코사인)
    V_norm = V / np.linalg.norm(V, axis=1, keepdims=True)
    H_norm = H / np.linalg.norm(H, axis=1, keepdims=True)
    sim_matrix = V_norm @ H_norm.T  # shape (N, M)

    # 4. Hungarian 알고리즘으로 최적 1:1 매칭
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    pairs = []
    for v_idx, h_idx in zip(row_ind, col_ind):
        score = float(sim_matrix[v_idx, h_idx])
        if score >= 0.89:  # 유사도 임계값 적용
            pairs.append({
                "vertical": os.path.join(vertical_path, vertical_files[v_idx]),
                "horizontal": os.path.join(horizontal_path, horizontal_files[h_idx]),
                "score": score
            })

    # 5. 결과 저장
    output_json = os.path.join(OUTPUT_DIR, f"{movie_name}.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(pairs)}개 페어가 `{output_json}`에 저장되었습니다.")

    # 6. 시각화 (옵션)
    if show_visualization and pairs:
        print(f"🖼️ 시각화 결과를 `{VISUALIZATION_DIR}/{movie_name}`에 저장합니다.")
        visualize_pairs(pairs, movie_name, show_visualization)

    # 7. 이미지 쌍 복사 (옵션)
    if save_pairs and pairs:
        copy_paired_images(pairs, movie_name)

def main():
    parser = argparse.ArgumentParser(description='Process image pairs using CLIP model')
    parser.add_argument('--show', action='store_true', help='Show visualization results')
    parser.add_argument('--save', action='store_true', help='Save paired images in folders')
    args = parser.parse_args()

    # vertical 디렉토리의 모든 영화 폴더를 순회
    for movie_name in sorted(os.listdir(VERTICAL_DIR)):
        # pairs 폴더가 있는 경우 건너뜀
        if os.path.exists(os.path.join(OUTPUT_DIR, movie_name)):
            print(f"⚠️ Skipping {movie_name}: Already processed")
            continue
        process_movie_pair(movie_name, args.show, args.save)

if __name__ == "__main__":
    main()