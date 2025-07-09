import os
from diffusers import DiffusionPipeline
import torch

# 1. Diffusion 모델 로드
print("모델 로드 중 . . . ")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16
).to("cuda")
print("모델 로드 완료.")

# 2. 기준 폴더 설정 (모든 하위 폴더 검색)
BASE_DIR = "."
TOTAL_IMAGES = 50  # 생성할 이미지 수

# 3. 모든 representative_caption.txt 탐색
for root, dirs, files in os.walk(BASE_DIR):
    if 'representative_caption.txt' in files:
        # caption 읽기
        caption_path = os.path.join(root, 'representative_caption.txt')
        with open(caption_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # 저장 경로 설정
        save_dir = os.path.join(root, 'createimage')

        # 이미 createimage 폴더가 존재하고 비어있지 않으면 건너뜀
        if os.path.exists(save_dir) and os.listdir(save_dir):
            print(f"[SKIP] {save_dir} 이미 생성되어 있음, 건너뜁니다.")
            continue

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n'{prompt}' → 이미지 생성 중: {save_dir}")

        # 이미지 생성 및 저장
        for i in range(TOTAL_IMAGES):
            image = pipe(prompt).images[0]
            filename = os.path.join(save_dir, f"image_{i+1:03d}.png")
            image.save(filename)
            print(f"  - [{i+1}/{TOTAL_IMAGES}] 저장됨: {filename}")