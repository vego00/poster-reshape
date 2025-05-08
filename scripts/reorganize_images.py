import os
import shutil
from tqdm import tqdm

# 설정
VERTICAL_DIR = "data/vertical"
HORIZONTAL_DIR = "data/horizontal"
IMAGES_DIR = "data/images"

def copy_images():
    # 모든 영화 디렉토리 가져오기
    vertical_movies = set(os.listdir(VERTICAL_DIR))
    horizontal_movies = set(os.listdir(HORIZONTAL_DIR))
    all_movies = vertical_movies.union(horizontal_movies)

    print(f"총 {len(all_movies)}개의 영화를 처리합니다.")

    # 각 영화별로 처리
    for movie_name in tqdm(all_movies, desc="Processing movies"):
        # 새로운 디렉토리 생성
        movie_dir = os.path.join(IMAGES_DIR, movie_name)
        os.makedirs(movie_dir, exist_ok=True)

        # vertical 이미지 복사
        vertical_path = os.path.join(VERTICAL_DIR, movie_name)
        if os.path.exists(vertical_path):
            for img in os.listdir(vertical_path):
                src = os.path.join(vertical_path, img)
                dst = os.path.join(movie_dir, f"v_{img}")
                shutil.copy2(src, dst)

        # horizontal 이미지 복사
        horizontal_path = os.path.join(HORIZONTAL_DIR, movie_name)
        if os.path.exists(horizontal_path):
            for img in os.listdir(horizontal_path):
                src = os.path.join(horizontal_path, img)
                dst = os.path.join(movie_dir, f"h_{img}")
                shutil.copy2(src, dst)

        print(f"✅ {movie_name}: 이미지 복사 완료")

if __name__ == "__main__":
    copy_images() 