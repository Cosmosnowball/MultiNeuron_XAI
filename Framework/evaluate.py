import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
import pandas as pd

# 사전학습된 ResNet18 모델과 Hook 설정
model = models.resnet18(pretrained=True)
model.eval()
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Hook 등록
model.conv1.register_forward_hook(get_activation('conv1'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 뉴런 인덱스 CSV에서 불러오기
def load_neuron_indices(csv_path):
    df = pd.read_csv(csv_path)
    indices = {}
    for _, row in df.iterrows():
        layer = row['Layer'].strip()
        if pd.isna(row['Selected Filter Indices']) or str(row['Selected Filter Indices']).strip() == "":
            continue
        idx_list = [int(i) for i in str(row['Selected Filter Indices']).split(',') if i.strip().isdigit()]
        if idx_list:
            indices[layer] = idx_list
    return indices

# 평균 활성화 계산
def compute_average_activations(folder_path, layer_neuron_indices):
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            _ = model(input_tensor)

        all_activations = []
        for layer_name, neuron_indices in layer_neuron_indices.items():
            layer_output = activation[layer_name].squeeze()  # (C, H, W)
            if layer_output.ndim == 3:
                pooled = layer_output.mean(dim=(1, 2))  # (C,)
            else:
                pooled = layer_output  # (C,)
            selected = pooled[neuron_indices]
            all_activations.append(selected.mean().item())

        avg_activation = np.mean(all_activations)
        results.append(avg_activation)

    return results

# AUC / MAD
def calculate_auc(A0, A1):
    labels = [0] * len(A0) + [1] * len(A1)
    scores = A0 + A1
    return roc_auc_score(labels, scores)

def calculate_mad(A0, A1):
    A0 = np.array(A0)
    A1 = np.array(A1)
    std_A0 = A0.std(ddof=1)
    if std_A0 == 0:
        return 0
    return (A1.mean() - A0.mean()) / std_A0

# 클러스터 평가
def evaluate_cluster(cluster_path, random_path):
    cluster_name = os.path.basename(cluster_path)
    csv_path = os.path.join(cluster_path, f"{cluster_name}.csv")
    createimage_path = os.path.join(cluster_path, "createimage")

    if not os.path.exists(csv_path) or not os.path.exists(createimage_path):
        print(f"누락: {csv_path} 또는 {createimage_path}")
        return

    neuron_indices = load_neuron_indices(csv_path)
    if not neuron_indices:
        print(f"뉴런 인덱스 없음: {csv_path}")
        return

    A1 = compute_average_activations(createimage_path, neuron_indices)
    A0 = compute_average_activations(random_path, neuron_indices)

    if not A1 or not A0:
        print(f"이미지 부족: {cluster_path}")
        return

    auc = calculate_auc(A0, A1)
    mad = calculate_mad(A0, A1)

    with open(os.path.join(cluster_path, "evaluation.txt"), 'w') as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"MAD: {mad:.4f}\n")

    print(f" {cluster_path}/evaluation.txt 저장 완료 (AUC={auc:.4f}, MAD={mad:.4f})")

# 폴더 재귀 탐색 및 평가 수행
def recursive_evaluate(root_path, random_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if "createimage" in dirnames:
            evaluate_cluster(dirpath, random_path)

# 실행
if __name__ == "__main__":
    random_path = "/Users/yunhyungnam/Downloads/a/random"  # 랜덤 경로
    top_cluster_root = "."  # 현재 디렉토리부터 재귀
    recursive_evaluate(top_cluster_root, random_path)
