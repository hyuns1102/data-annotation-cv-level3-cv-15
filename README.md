# Semantic Segmentation (재활용 쓰레기 분류)

## Table of Contents

- [Background](#background)
- [Usage](#usage)
  - [Structure](#Structure)
  - [Requirements](#install)
  - [Getting_Started](#Getting_Started)
- [Result](#Result)

## Background

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

![img](https://github.com/boostcampaitech2/data-annotation-cv-level3-cv-15/blob/master/img/img1.png)
(출처 : 위키피디아)

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.

예측 csv 파일 제출 (Evaluation) 방식이 아닌 model checkpoint 와 inference.py 를 제출하여 채점하는 방식입니다. (Inference) 상세 제출 방법은 AI Stages 가이드 문서를 참고해 주세요!

대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 베이스라인 코드 탭 하단의 설명을 참고해주세요.

Input : 글자가 포함된 전체 이미지

Output : bbox 좌표가 포함된 UFO Format (상세 제출 포맷은 평가 방법 탭 및 강의 5강 참조)

## Usage
### Structure
```sh
|-- README.md
|-- code
|   |-- convert_mlt.py
|   |-- dataset.py
|   |-- detect.py
|   |-- deteval.py
|   |-- east_dataset.py
|   |-- inference.py
|   |-- loss.py
|   |-- model.py
|   `-- train.py
`-- notebook
    |-- EDA # Noise Image & Error Image
    |-- office_hours # office hour
    `-- utils # k-fold & SWA & Normalize
    
# 서버내 input 경로
└─ input
    └─ data
        └─ ICDAR2017_Korean
            └─ ufo
            │    └─ train.json
            └─ images
                 │ img_1001.jpg
                 │ img_1002.jpg
                 │ img_1003.jpg
                 │ ...
                 │ img_4700.jpg
```
### Install

- Requirements

    ```sh
    $ pip install -r requirements.txt
    ```

### Getting_Started

- Custom_Baseline_Code

    ```sh
    $ cd code
    $ python train.py

    [Parameter setting : args]
    ```




## Result

- 학습이 완료되면 자동으로 trained_models 경로에 latest.pth 모델이 저장됩니다.
- Save Server and Submit


## How to Evaluate Result

```
평가방법은 7강에서 소개되는 DetEval 방식으로 계산되어 진행됩니다.

DetEval은, 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법 중 하나 입니다.

평가가 이루어지는 방법은 다음과 같습니다.
```

1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산해냅니다.

2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다.

3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정됩니다.


## Reference

Wolf, C., & Jolion, J. M. (2006). Object count/area graphs for the evaluation of object detection and segmentation algorithms. (IJDAR), 8(4), 280-296.
