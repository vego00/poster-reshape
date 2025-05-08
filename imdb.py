import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("IMDB_API_KEY")
base_url = "https://imdb-api.com/en/API"
IMAGE_BASE_URL = "https://imdb-api.com/images/original"

def download_image(url, save_dir, prefix):
    if not url:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(save_dir, prefix + "_" + filename)

    try:
        img_data = requests.get(url).content
        with open(save_path, "wb") as f:
            f.write(img_data)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

# 1) 인기 영화 100개 수집
popular_movies = []
params = {
    "apiKey": API_KEY
}

# IMDb API를 사용하여 인기 영화 목록 가져오기
resp = requests.get(f"{base_url}/MostPopularMovies", params=params)
data = resp.json()
popular_movies.extend(data.get("items", []))

print(f"Collected {len(popular_movies)} popular movies")

# 2) 각 영화의 ID로 이미지 정보 조회 및 다운로드
for movie in popular_movies:
    movie_id = movie["id"]
    movie_title = movie.get("title", "unknown")

    # 파일명에 사용할 수 없는 문자를 _로 대체
    for m in movie_title:
        if m in ' /\\:*?"<>|':
            movie_title = movie_title.replace(m, '_')

    # 영화 상세 정보 가져오기
    params = {
        "apiKey": API_KEY
    }
    resp = requests.get(f"{base_url}/Title/{API_KEY}/{movie_id}", params=params)
    movie_data = resp.json()

    # 포스터 이미지 URL 가져오기
    poster_url = movie_data.get("image")
    
    if poster_url:
        download_image(
            poster_url,
            save_dir=f"data/vertical/{movie_title}",
            prefix=f"{movie_id}_poster"
        ) 