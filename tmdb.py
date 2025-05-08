import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
base_url = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"

def download_image(file_path, save_dir, prefix, max_retries=3):
    url = IMAGE_BASE_URL + file_path
    os.makedirs(save_dir, exist_ok=True)
    filename = file_path.split("/")[-1]
    save_path = os.path.join(save_dir, prefix + "_" + filename)

    for attempt in range(max_retries):
        try:
            img_data = requests.get(url, timeout=10).content
            with open(save_path, "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {save_path}")
            return
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download {url} after {max_retries} attempts: {str(e)}")

# 1) 인기 영화 40개 수집
popular_movies = []
for page in range(2, 3):  # 5페이지까지 수집 (페이지당 20개씩)
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": page
    }
    try:
        resp = requests.get(f"{base_url}/movie/popular", params=params, timeout=10)
        resp.raise_for_status()  # HTTP 오류 확인
        data = resp.json()
        popular_movies.extend(data.get("results", []))
        print(f"Collected page {page}: {len(data.get('results', []))} movies")
    except requests.exceptions.RequestException as e:
        print(f"Error collecting page {page}: {str(e)}")
        continue

print(f"Total collected {len(popular_movies)} popular movies")

# 2) 각 영화의 ID로 이미지 정보 조회 및 다운로드
for movie in popular_movies:
    movie_id = movie["id"]
    movie_title = movie.get("title", "unknown")

    # 파일명에 사용할 수 없는 문자를 _로 대체
    for m in movie_title:
        if m in ' /\\:*?"<>|':
            movie_title = movie_title.replace(m, '_')

    # 이미지 정보 가져오기
    params = {
        "api_key": API_KEY,
        "language": "en-US,ko-KR",
        "include_image_language": "en,ko,null"
    }
    
    try:
        resp = requests.get(f"{base_url}/movie/{movie_id}/images", params=params, timeout=10)
        resp.raise_for_status()
        img_data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting images for movie {movie_title}: {str(e)}")
        continue

    posters = img_data.get("posters", [])
    backdrops = img_data.get("backdrops", [])

    if posters:
        for i, poster in enumerate(posters):
            download_image(
                poster["file_path"],
                save_dir=f"data/vertical/{movie_title}",
                prefix=f"{movie_id}_poster_{i}"
            )

    if backdrops:
        for i, backdrop in enumerate(backdrops):
            download_image(
                backdrop["file_path"],
                save_dir=f"data/horizontal/{movie_title}",
                prefix=f"{movie_id}_backdrop_{i}"
            )