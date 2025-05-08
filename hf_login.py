import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))