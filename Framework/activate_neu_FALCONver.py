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

# 각 이미지 폴더 내의 모든 이미지 파일을 처리하고 활성화 필터를 집계하여 CSV로 저장
def process_images_and_save_filters(folder_path):
    image_paths = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(folder_path, fname))

    if not image_paths:
        print(f"{folder_path} → 이미지가 없습니다.")
        return False

    filter_counts = {layer: None for layer in ['conv1','layer1','layer2','layer3','layer4']}

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"{img_path} 불러오기 실패: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0)
            _ = model(input_tensor)

            if idx == 0:
                for layer_name, activation in activations.items():
                    filter_counts[layer_name] = np.zeros(activation.shape[1], dtype=int)

            for layer_name, activation in activations.items():
                # 채널별 평균 활성화 계산
                mean_per_filter = activation.mean(dim=(0, 2, 3)).cpu().numpy()
                # Falcon 방식으로 threshold 계산
                m = mean_per_filter.mean()      # 전체 채널 평균
                s = mean_per_filter.std()       # 전체 채널 표준편차
                lim = 2                         # Falcon 하이퍼파라미터
                high_thresh = m + lim * s       # 상위 활성화 기준
                selected_filters = np.where(mean_per_filter > high_thresh)[0]

                for f_idx in selected_filters:
                    filter_counts[layer_name][f_idx] += 1

    # 절반 이상의 이미지에서 나타난 필터를 선택
    half_count = len(image_paths) / 2
    results = []
    for layer_name, counts in filter_counts.items():
        sel = np.where(counts >= half_count)[0].tolist()
        results.append({'layer': layer_name, 'selected_filters': sel})

    current_dir = os.path.basename(folder_path)
    csv_name = f"{current_dir}.csv"
    csv_path = os.path.join(folder_path, csv_name)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Layer','Selected Filter Indices'])
        for r in results:
            writer.writerow([r['layer'], ', '.join(map(str, r['selected_filters']))])

    print(f"{folder_path} → {csv_name} 저장 완료")
    return True

# 모든 하위 폴더에서 공통 활성화 뉴런 필터를 집계하여 상위 폴더에 CSV로 저장
def aggregate_common_filters_from_children(parent_path):
    child_folders = [
        os.path.join(parent_path, name)
        for name in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, name))
           and name.lower() not in ("createimage","samples")
    ]

    csv_files = []
    for f in child_folders:
        for fname in os.listdir(f):
            if fname.endswith(".csv"):
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
            if pd.notna(filters_str) and isinstance(filters_str, str) and filters_str.strip():
                neuron_indices = set(map(int, filters_str.split(',')))
            else:
                neuron_indices = set()
            layer_neuron_sets.setdefault(layer, []).append(neuron_indices)

    common_neurons_per_layer = {
        layer: sorted(set.intersection(*sets)) if sets else []
        for layer, sets in layer_neuron_sets.items()
    }

    parent_name = os.path.basename(os.path.normpath(parent_path)) or "root"
    out_path = os.path.join(parent_path, f"{parent_name}.csv")
    pd.DataFrame([
        {'Layer': layer, 'Selected Filter Indices': ', '.join(map(str, inds))}
        for layer, inds in common_neurons_per_layer.items()
    ]).to_csv(out_path, index=False, encoding='utf-8')

    print(f"{parent_path} → {parent_name}.csv 저장 완료")
    return True

# 재귀 탐색 함수
def recursive_traverse(folder_path, level=0):
    base = os.path.basename(folder_path).lower()
    if base in ("createimage","samples"):
        print(f"[SKIP] '{base}' 폴더는 처리하지 않습니다: {folder_path}")
        return

    subfolders = [
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
           and f.lower() not in ("createimage","samples")
    ]
    full_paths = [os.path.join(folder_path, sub) for sub in subfolders]

    if not full_paths:
        process_images_and_save_filters(folder_path)
    else:
        for sub in full_paths:
            recursive_traverse(sub, level+1)

        imgs = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg','.png','.jpeg'))
                   and not os.path.isdir(os.path.join(folder_path, f))]
        if imgs:
            process_images_and_save_filters(folder_path)

        if folder_path != TOP_FOLDER:
            aggregate_common_filters_from_children(folder_path)

if __name__ == "__main__":
    print("공통 활성화 뉴런 필터 집계 시작...")
    recursive_traverse(TOP_FOLDER)
