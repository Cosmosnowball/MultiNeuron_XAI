import os
import shutil
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt '''시각화 부분 버전 충돌로 주석처리'''
#from sklearn.manifold import TSNE '''시각화 부분 버전 충돌로 주석처리'''
from sentence_transformers import SentenceTransformer
from finch import FINCH  # finch.py가 같은 폴더에 있어야 함

# 1. 데이터 로드
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 2. 텍스트 임베딩 생성
def create_embeddings(captions, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(captions)
    return embeddings

# 3. FINCH 클러스터링 적용
def apply_finch_clustering(embeddings):
    c, num_clust, req_c = FINCH(embeddings)
    return c

# 4. t-SNE로 임베딩 시각화
'''시각화 부분 버전 충돌로 주석처리'''
'''
def reduce_dimension(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d
'''

# 5. 결과 저장 및 시각화
def save_clusters_and_visualizations(df, clustering_results, output_base='/local_datasets/FINCH_result'):
    os.makedirs(output_base, exist_ok=True)

    for level in range(clustering_results.shape[1]):
        cluster_labels = clustering_results[:, level]

        # 클러스터 결과 저장
        df_copy = df.copy()
        df_copy['cluster'] = cluster_labels
        csv_path = os.path.join(output_base, f'clustered_captions_level_{level+1}.csv')
        df_copy.to_csv(csv_path, index=False)

        # 시각화 저장
        '''시각화 부분 버전 충돌로 주석처리'''
        '''
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='tab20', s=15)
        plt.title(f'FINCH Clustering Visualization (Level {level+1})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, label='Cluster ID')
        plot_path = os.path.join(output_base, f'tsne_plot_level_{level+1}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        '''

# 6. 계층구조로 이미지 정리
def organize_images_by_clusters(result_folder, image_folder='unsplash_images', output_base_folder='FINCH_clustered_images'):
    result_files = sorted([f for f in os.listdir(result_folder) if f.startswith('clustered_captions_level_') and f.endswith('.csv')])
    dfs = [pd.read_csv(os.path.join(result_folder, file)) for file in result_files]

    df = dfs[0][['Image']].copy()
    for level_idx, level_df in enumerate(dfs):
        df[f'level_{level_idx+1}_cluster'] = level_df['cluster']

    for idx, row in df.iterrows():
        image_name = row['Image']
        folder_path = output_base_folder
        for level_idx in reversed(range(1, len(dfs) + 1)):
            cluster_id = row[f'level_{level_idx}_cluster']
            folder_path = os.path.join(folder_path, f'level{level_idx}_cluster_{cluster_id}')
            os.makedirs(folder_path, exist_ok=True)

        src_path = os.path.join(image_folder, image_name)
        dst_path = os.path.join(folder_path, image_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f" Warning: {src_path} 파일이 존재하지 않음. 건너뜀.")

    print("이미지를 계층적 폴더로 정리 완료")

# 메인 실행부
def main():
    # 파일 경로
    caption_file = 'output_explain.csv' #참고할 캡션 파일, 같은 디렉토리 안에 있어야 함
    image_folder = '/local_datasets/ImageNet_val' #원본 이미지 폴더 경로

    # 1. 데이터 불러오기
    df = load_data(caption_file)
    captions = df['Caption'].tolist()

    # 2. 임베딩 생성
    embeddings = create_embeddings(captions)

    # 3. FINCH 클러스터링
    clustering_results = apply_finch_clustering(embeddings)

    # 4. 차원 축소 (t-SNE)
    '''시각화 부분 버전 충돌로 주석처리'''
    #embeddings_2d = reduce_dimension(embeddings)

    # 5. 결과 저장 및 시각화
    #save_clusters_and_visualizations(df, clustering_results, embeddings_2d) 아랫줄 코드로 대체, 시각화 주석처리.
    save_clusters_and_visualizations(df, clustering_results)

    # 6. 이미지 계층 폴더 정리
    organize_images_by_clusters(
        result_folder='/local_datasets/FINCH_result',
        image_folder=image_folder,
        output_base_folder='/local_datasets/ImageNet_val_FINCH'
    )

if __name__ == '__main__':
    main()
