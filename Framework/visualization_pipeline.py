import os
import random
import json
import csv
from PIL import Image, ImageDraw, ImageFont


def get_cluster_label(dir_name):
    import re
    m = re.match(r'level(\d+)_cluster_(\d+)', dir_name)
    if m:
        level, cluster = m.groups()
        return f"L{level}_{cluster}"
    return dir_name


def make_montage(image_paths, caption, neuron_info, output_path,
                 thumb_size=(128, 128), grid_size=(5, 2), margin=5, font_path=None):
    cols, rows = grid_size
    thumb_w, thumb_h = thumb_size

    # 폰트 로드
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=16)
    else:
        font = ImageFont.load_default()

    # 텍스트 크기 계산: Pillow v10 이후 textsize 대신 getbbox 사용
    # 대표 캡션 높이
    cap_bbox = font.getbbox(caption)
    cap_h = cap_bbox[3] - cap_bbox[1]
    # 뉴런 정보 높이 합
    info_lines = neuron_info.split('\n')
    info_h = 0
    for line in info_lines:
        bbox = font.getbbox(line)
        line_h = bbox[3] - bbox[1]
        info_h += line_h + margin

    text_area_h = cap_h + margin + info_h

    # 몽타주 캔버스 크기
    montage_w = cols * (thumb_w + margin) + margin
    montage_h = rows * (thumb_h + margin) + margin + text_area_h + margin
    montage = Image.new('RGB', (montage_w, montage_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)

    # 썸네일 붙이기
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img.thumbnail((thumb_w, thumb_h))
        x = margin + (i % cols) * (thumb_w + margin)
        y = margin + (i // cols) * (thumb_h + margin)
        montage.paste(img, (x, y))

    # 텍스트 출력 좌표
    text_x = margin
    text_y = margin + rows * (thumb_h + margin) + margin

    # 대표 캡션 그리기
    draw.text((text_x, text_y), caption, font=font, fill=(0, 0, 0))

    # 뉴런 정보 그리기
    y_offset = text_y + cap_h + margin
    for line in info_lines:
        draw.text((text_x, y_offset), line, font=font, fill=(0, 0, 0))
        bbox = font.getbbox(line)
        line_h = bbox[3] - bbox[1]
        y_offset += line_h + margin

    montage.save(output_path)


def process_directory(dir_path, base_path, thumb_size=(128, 128)):
    # 'samples' 폴더는 제외
    children = [d for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d)) and d != 'samples']
    samples_dir = os.path.join(dir_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    sampled_list = []

    if children:
        for child in children:
            child_json = os.path.join(dir_path, child, 'sampled_images.json')
            if os.path.exists(child_json):
                with open(child_json, 'r') as f:
                    sampled_list.extend(json.load(f))
    else:
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(valid_exts):
                sampled_list.append(os.path.join(dir_path, fname))

    if not sampled_list:
        print(f"{dir_path}에 이미지가 없어 건너뜁니다.")
        return

    sampled = random.sample(sampled_list, min(10, len(sampled_list)))
    if len(sampled) < 10:
        sampled += [sampled[-1]] * (10 - len(sampled))

    downsized_paths = []
    for idx, img_path in enumerate(sampled):
        try:
            img = Image.open(img_path)
            img.thumbnail(thumb_size)
            thumb_path = os.path.join(samples_dir, f'sample_{idx}.png')
            img.save(thumb_path)
            downsized_paths.append(thumb_path)
        except Exception as e:
            print(f"이미지 처리 실패: {img_path} -> {e}")

    with open(os.path.join(dir_path, 'sampled_images.json'), 'w') as f:
        json.dump(downsized_paths, f)

    cap_file = os.path.join(dir_path, 'representative_caption.txt')
    caption = open(cap_file, 'r', encoding='utf-8').read().strip() if os.path.exists(cap_file) else ''

    csv_file = os.path.join(dir_path, os.path.basename(dir_path) + '.csv')
    neuron_info = ''
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            neuron_info = '\n'.join([', '.join(row) for row in reader])

    rel_parts = os.path.relpath(dir_path, base_path).split(os.sep)
    montage_name = '-'.join(get_cluster_label(p) for p in rel_parts) + '.png'
    viz_dir = os.path.join(base_path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(viz_dir, montage_name)

    make_montage(downsized_paths, caption, neuron_info, output_path)
    print(f"{dir_path} 처리가 완료되어 {output_path}에 저장되었습니다.")


def main():
    base_path = '/local_datasets/ImageNet_val_FINCH'
    for root, dirs, files in os.walk(base_path, topdown=False):
        if 'representative_caption.txt' in files:
            process_directory(root, base_path)

if __name__ == '__main__':
    main()
