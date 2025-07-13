import os
import csv
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd

TOP_FOLDER = "/local_datasets/ImageNet_val_FINCH"  # FINCH 클러스터링 결과 폴더

# ResNet18 사전학습 모델
model = models.resnet18(pretrained=True)
model.eval()

# 중간 레이어 출력 저장
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Hook 등록
model.conv1.register_forward_hook(save_activation('conv1'))
model.layer1.register_forward_hook(save_activation('layer1'))
model.layer2.register_forward_hook(save_activation('layer2'))
model.layer3.register_forward_hook(save_activation('layer3'))
model.layer4.register_forward_hook(save_activation('layer4'))

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 각 이미지 폴더(레벨 1, 최하위 클러스터) 내의 모든 이미지 파일을 처리하고 활성화 필터를 집계하여 CSV로 저장
def process_images_and_save_filters(folder_path):
    #파일 내 이미지 경로 수집
    image_paths = []
    
    #폴더 내 모든 파일 순회, 이미지 파일만 수집
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):  # 이미지 확장자 필터링
            image_paths.append(os.path.join(folder_path, fname)) #파일의 전체 경로를 저장장

    if not image_paths:
        print(f"{folder_path} → 이미지가 없습니다.")
        return False  # 이미지 없음

    filter_counts = {layer: None for layer in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']}

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"{img_path} 불러오기 실패: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0) #전처리 수행후 배치 차원 추가
            _ = model(input_tensor) #모델에 입력하여 활성화 저장

            #     첫 번째 이미지 처리 시에만, activations에 저장된 레이어별 채널 수 확인 후
            #     filter_counts의 None 자리에 0으로 초기화된 배열 할당
            if idx == 0:
                for layer_name, activation in activations.items():
                    num_filters = activation.shape[1]
                    filter_counts[layer_name] = np.zeros(num_filters, dtype=int)

            #   저장된 activations를 레이어별로 순회
            for layer_name, activation in activations.items():
                mean_per_filter = activation.mean(dim=(0, 2, 3)).cpu().numpy()  # 채널별 평균 활성화 계산
                threshold = np.percentile(mean_per_filter, 90)                  # 상위 10% 필터 선택
                selected_filters = np.where(mean_per_filter > threshold)[0]     # 선택된 필터 인덱스

                # 선택된 필터 인덱스 마다 filter_counts 1씩 증가
                for f_idx in selected_filters:
                    filter_counts[layer_name][f_idx] += 1

    # ───────────────────────────────────────────────────────────────────
    # 이미지 절반 이상에서 활성화된 필터만 최종 선정
    half_count = len(image_paths) / 2
    results = []
    for layer_name, counts in filter_counts.items():
        selected_indices = np.where(counts >= half_count)[0].tolist()
        results.append({'layer': layer_name, 'selected_filters': selected_indices})

    parent_dir = os.path.basename(os.path.dirname(folder_path))
    current_dir = os.path.basename(folder_path)

    # csv 파일 이름 생성
    csv_name = f"{current_dir}.csv"

    csv_filename = os.path.join(folder_path, csv_name) # 최종 CSV 파일 경로
 
    # CSV 파일 작성 
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Layer', 'Selected Filter Indices'])
        for r in results:
            writer.writerow([r['layer'], ', '.join(map(str, r['selected_filters']))])

    print(f"{folder_path} → {csv_name} 저장 완료")

    return True # 이미지 처리 및 필터 집계 성공

# 모든 하위 폴더에서 공통 활성화 뉴런 필터를 집계하여 상위 폴더에 CSV로 저장
def aggregate_common_filters_from_children(parent_path):
    child_folders = [
        os.path.join(parent_path, name)
        for name in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, name))
           and name.lower() not in ("createimage", "samples")
    ]

    csv_files = []
    for f in child_folders:
        for fname in os.listdir(f):
            if fname.endswith(".csv") and not fname.endswith("_createimage.csv"):
                csv_files.append(os.path.join(f, fname))
                break

    if not csv_files:
        return False

    layer_neuron_sets = {}
    for file in csv_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            layer = row['Layer']
            filters_str = row['Selected Filter Indices']

            # 항상 set을 넣되, 비어있으면 빈 set
            neuron_indices = set(map(int, filters_str.split(','))) if pd.notna(filters_str) and filters_str.strip() else set()

            if layer not in layer_neuron_sets:
                layer_neuron_sets[layer] = []

            layer_neuron_sets[layer].append(neuron_indices)


    # 공통 필터 계산
    common_neurons_per_layer = {
        layer: sorted(set.intersection(*sets)) if sets else []
        for layer, sets in layer_neuron_sets.items()
    }

    result_df = pd.DataFrame([
        {'Layer': layer, 'Selected Filter Indices': ', '.join(map(str, indices))}
        for layer, indices in common_neurons_per_layer.items()
    ])

    parent_name = os.path.basename(os.path.normpath(parent_path)) or "root"
    output_path = os.path.join(parent_path, f"{parent_name}.csv")

    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"{parent_path} → {parent_name}.csv 저장 완료")
    return True

def recursive_traverse(folder_path, level=0):
    subfolders = [
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
           and f.lower() not in ("createimage", "samples")
    ]
    full_paths = [os.path.join(folder_path, sub) for sub in subfolders]
    
    
    # 'createimage' 또는 'samples' 폴더는 완전 무시
    if os.path.basename(folder_path).lower() in ("createimage", "samples"):
        print(f"[SKIP] createimage 또는 samples 폴더는 처리하지 않습니다: {folder_path}")
        return

    # 하위 폴더가 없다면 (leaf 폴더): 이미지 처리
    if not full_paths:
        process_images_and_save_filters(folder_path)
    else:
        # 하위 폴더들 재귀 순회
        for sub in full_paths:
            recursive_traverse(sub, level + 1)

        # 상위 폴더 처리: createimage 폴더는 이미지 처리 대상에서 제외
        # → createimage를 제외한 현재 폴더의 이미지만 처리
        files_in_current = os.listdir(folder_path)
        image_files = [
            f for f in files_in_current
            if f.lower().endswith(('.jpg', '.png'))
            and f != 'createimage'
            and not os.path.isdir(os.path.join(folder_path, f))
        ]

        if image_files:
            process_images_and_save_filters(folder_path)

        # 공통 뉴런 집계: 최상위 폴더(TOP_FOLDER)에서는 스킵
        if folder_path != TOP_FOLDER:
            aggregate_common_filters_from_children(folder_path)


# 최상위 경로부터 실행 (예: level6이 있는 경로)
if __name__ == "__main__":
    print("공통 활성화 뉴런 필터 집계 시작...")
    top_folder = TOP_FOLDER # FINCH 클러스터링 결과 폴더
    recursive_traverse(top_folder)

