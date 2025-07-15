import os
import random
import json
import csv
import textwrap
from PIL import Image, ImageDraw, ImageFont


def get_cluster_label(dir_name):
    """
    폴더 이름(levelX_cluster_Y)에서 레벨과 클러스터 번호를 추출해 'Lx_Y' 형식으로 반환합니다.
    예: 'level1_cluster_42' -> 'L1_42'
    """
    import re
    m = re.match(r'level(\d+)_cluster_(\d+)', dir_name)
    if m:
        level, cluster = m.groups()
        return f"L{level}_{cluster}"
    return dir_name  # 패턴에 맞지 않으면 원본 반환


def wrap_text(text, font, max_width):
    """
    긴 텍스트를 지정된 너비(max_width)에 맞춰 단어 단위로 줄바꿈해 리스트로 반환합니다.
    """
    words = text.split()
    if not words:
        return ['']
    lines = []
    current = words[0]
    for word in words[1:]:
        test = current + ' ' + word
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test  # 현재 줄에 단어 추가
        else:
            lines.append(current)  # 줄이 넘치면 개행
            current = word
    lines.append(current)
    return lines


def make_montage(
    image_paths,       # 썸네일로 사용할 이미지 경로 리스트
    caption,           # 대표 캡션(자연어 문장)
    neuron_info,       # CSV로부터 읽어온 뉴런 정보 문자열(줄바꿈 포함)
    output_path,       # 최종 저장할 파일 경로(확장자 .png로 지정)
    thumb_size=(64, 64),
    grid_size=(5, 2),
    margin=3,
    font_path=None     # 외부 폰트를 사용할 경우 경로 지정
):
    # ───── 타이틀(클러스터 레이블) 설정 ─────
    title_text = os.path.basename(output_path).replace('.png', '')

    cols, rows = grid_size
    thumb_w, thumb_h = thumb_size

    # ───── 폰트 로드 ─────
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=12)
    else:
        font = ImageFont.load_default()

    # ───── 레이아웃 계산 ─────
    montage_w = cols * (thumb_w + margin) + margin        # 전체 너비
    max_text_w = montage_w - 2 * margin                  # 텍스트 영역 최대 너비

    # 캡션과 뉴런 정보 줄바꿈 처리
    cap_lines = wrap_text(caption, font, max_text_w)
    raw_info = neuron_info.split('\n') if neuron_info else []
    info_lines = []
    for line in raw_info:
        info_lines.extend(wrap_text(line, font, max_text_w))

    # 텍스트 높이 계산
    tbbox = font.getbbox(title_text)
    title_h = tbbox[3] - tbbox[1]

    cap_h_total = sum((font.getbbox(line)[3] - font.getbbox(line)[1]) for line in cap_lines)
    cap_h_total += (len(cap_lines) - 1) * margin
    info_h_total = sum((font.getbbox(line)[3] - font.getbbox(line)[1]) + margin for line in info_lines)

    text_area_h = title_h + margin + cap_h_total + margin + info_h_total

    montage_h = rows * (thumb_h + margin) + margin + text_area_h + margin  # 전체 높이

    # ───── 캔버스 생성 ─────
    montage = Image.new('RGB', (montage_w, montage_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)

    # ───── 타이틀 그리기 ─────
    text_x = margin
    text_y = margin
    draw.text((text_x, text_y), title_text, font=font, fill=(0, 0, 0))

    # ───── 썸네일 배치 ─────
    thumb_start_y = text_y + title_h + margin
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            img.thumbnail((thumb_w, thumb_h))
            x = margin + (i % cols) * (thumb_w + margin)
            y = thumb_start_y + (i // cols) * (thumb_h + margin)
            montage.paste(img, (x, y))
        except Exception:
            # 누락된 파일 등 예외 시 무시
            pass

    # ───── 캡션 그리기 ─────
    cap_start_y = thumb_start_y + rows * (thumb_h + margin) + margin
    y_offset = cap_start_y
    for line in cap_lines:
        draw.text((text_x, y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + margin
    y_offset += margin

    # ───── 뉴런 정보 그리기 ─────
    for line in info_lines:
        draw.text((text_x, y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + margin

    # ───── 결과 저장(WebP 압축) ─────
    webp_path = output_path.replace('.png', '.webp')
    montage.save(
        webp_path,
        format='WEBP',
        quality=50,
        method=6,
        optimize=True
    )


def process_directory(dir_path, base_path, thumb_size=(64, 64)):
    """
    단일 클러스터 디렉토리 처리:
      1) 이전 samples 폴더 초기화
      2) 리프/내부 구분하여 이미지 경로 수집
      3) 최대 10개의 썸네일 생성
      4) representative_caption.txt 와 .csv 읽기
      5) make_montage 호출 후 결과 저장
    """
    # 1) 'samples' 폴더 초기화
    samples_dir = os.path.join(dir_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    for f in os.listdir(samples_dir):
        try:
            os.remove(os.path.join(samples_dir, f))
        except Exception:
            pass

    # 2) 하위 폴더가 있으면 자식 samples.json 사용, 없으면 원본 이미지 사용
    children = [d for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d)) and d != 'samples']
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

    # 이미지 없으면 건너뜀
    if not sampled_list:
        print(f"{dir_path}에 이미지가 없어 건너뜁니다.")
        return

    # 3) 랜덤 샘플링 (최대 10개)
    sampled = random.sample(sampled_list, min(10, len(sampled_list)))

    # 4) 썸네일 생성 및 JSON 기록
    downsized_paths = []
    for idx, img_path in enumerate(sampled):
        try:
            img = Image.open(img_path)
            img.thumbnail(thumb_size)
            thumb_path = os.path.join(samples_dir, f'sample_{idx}.png')
            img.save(thumb_path, format='PNG', optimize=True)
            downsized_paths.append(thumb_path)
        except Exception:
            pass
    with open(os.path.join(dir_path, 'sampled_images.json'), 'w') as f:
        json.dump(downsized_paths, f)

    # 5) 캡션, 뉴런 정보 로딩
    cap_file = os.path.join(dir_path, 'representative_caption.txt')
    caption = open(cap_file, 'r', encoding='utf-8').read().strip() if os.path.exists(cap_file) else ''
    csv_file = os.path.join(dir_path, os.path.basename(dir_path) + '.csv')
    neuron_info = ''
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            neuron_info = '\n'.join([', '.join(row) for row in reader])

    # 몽타주 생성 경로 설정 및 호출
    rel_parts = os.path.relpath(dir_path, base_path).split(os.sep)
    montage_name = '-'.join(get_cluster_label(p) for p in rel_parts) + '.png'
    viz_dir = os.path.join(base_path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(viz_dir, montage_name)
    make_montage(downsized_paths, caption, neuron_info, output_path)

    print(f"{dir_path} 처리가 완료되어 {output_path.replace('.png', '.webp')}에 저장되었습니다.")


def main():
    # 최상위 폴더 설정
    base_path = '/local_datasets/ImageNet_val_FINCH'
    for root, dirs, files in os.walk(base_path, topdown=False):
        if 'representative_caption.txt' in files:
            process_directory(root, base_path)


if __name__ == '__main__':
    main()
