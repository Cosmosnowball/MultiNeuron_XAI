01. generate_captions.py
모든 이미지에 대한 캡션을 생성. csv 파일 생성.

02. FINCH_clustering.py
생성된 캡션을 기반으로 이미지를 클러스터링하는 코드.
FINCH 클러스터링 결과 폴더 생성(FINCH_result : 클러스터링 결과 csv 파일 + 클러스터링 시각 자료). 계층적으로 클러스터링된 폴더 구조 생성(ImageNet_val_FINCH).

+ finch.py 
FINCH_clustering.py를 실행하는데 필요한 finch 모듈.

------

03. activate_neu
공통 활성화 뉴런 그룹을 구하는 코드. 
finch 클러스터링 결과 맨 상위 레벨의 디렉토리에서 실행 -> 각 클러스터마다 공통 활성과 뉴런 그룹을 구해서 {레벨_클러스터번호}.csv 로 저장함.

04. represent
대표 캡션을 추출하는 코드.
finch 클러스터링 결과 맨 상위 레벨의 디렉토리에서 실행 + 같은 디렉토리에 output_explain.csv 필요 (모든 이미지에 BLIP 적용한 결과 파일) -> 각 클러스터마다 대표 캡션을 txt 파일로 저장함. (representative_caption)  #현재 openai key 사용중. 이미지 25,000장 전체 대상으로 돌리면 1달러 미만 지출하는 듯함.

------

05. generate_images
평가 코드를 실행하기 위해 stable diffusion으로 이미지셋 생성하는 코드 (GPU 사용 필수. 굉장히 오래걸림)
finch 클러스터링 결과 맨 상위 레벨의 디렉토리에서 실행 -> 각 클러스터의 representative_caption(대표캡션)을 읽어서 각각에 맞는 이미지셋을 생성.

06. evaluate
CoSy 기반 평가 코드.
finch 클러스터링 결과 맨 상위 레벨의 디렉토리에서 실행 + generated_images폴더 필요 + random 이미지 폴더 필요(CoSy 평가기반. 사용한 이미지 데이터셋에서 아무거나 랜덤으로 뽑아서 넣어놓으면 됨) -> 평가값 생성