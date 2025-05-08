from datasets import Dataset
import json
import os
from PIL import Image
from datasets import Features, Value, ClassLabel
from huggingface_hub import login

# Step 1: Hugging Face 로그인
login(token=os.getenv("HF_TOKEN"))  # .env 파일에서 HF_TOKEN 읽음

# Step 2: JSON 로드
with open("data/pairs.json", "r") as f:
    data = json.load(f)

# Step 3: 이미지 경로 보정
for item in data:
    item["vertical"] = os.path.abspath(os.path.join("data/vertical", item["vertical"]))
    item["horizontal"] = os.path.abspath(os.path.join("data/horizontal", item["horizontal"]))

# Step 4: Dataset 생성
dataset = Dataset.from_list(data)

# Step 5: Push to Hub
dataset.push_to_hub("your-username/movie-poster-pairs", private=True)
