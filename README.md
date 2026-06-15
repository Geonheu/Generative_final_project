# c-MAS 기반 생성형 가상 멀티뷰 증강을 통한 행동 인식 성능 향상 연구

NTU60 단일 카메라 뷰 골격 데이터에 대해 c-MAS (Cross-view Motion Aware Synthesis) 디퓨전 모델로 가상 멀티뷰를 생성하고, MotionBERT 백본 위에서 다양한 융합 전략을 통해 행동 인식 정확도를 높이는 실험을 정리한 프로젝트.

---

## 연구 개요

### 문제 정의

실제 환경에서는 단일 카메라로만 행동을 포착하는 경우가 많아 시점 다양성이 부족하다. 멀티뷰 데이터를 확보하려면 카메라를 추가로 설치해야 하지만, c-MAS 디퓨전 모델을 활용하면 단일 뷰 입력으로부터 가상 멀티뷰 골격을 생성해 이 문제를 해결할 수 있다.

### 파이프라인 비교

두 가지 파이프라인을 대조해 생성형 가상 멀티뷰의 효용성과 한계를 검증했다.

| | GT 파이프라인 (상한선) | c-MAS 파이프라인 |
|---|---|---|
| 입력 | Kinect-25 3D | Kinect-25 3D |
| 변환 경로 | Kinect-25 → COCO-17 3D → 회전 → 2D | Kinect-25 → NBA-16 2D → **c-MAS** → NBA-16 3D → COCO-17 → 회전 → 2D |
| 특징 | 이상적 기하학적 정답, 노이즈 없음 | 도메인 변환 2회 + 확률적 생성, 오차 누적 가능 |
| 데이터 규모 | Train 40,091 / Val 16,487 샘플 | Val 480 샘플 (Zero-shot) |

---

## 실험 결과

### 실험 2: GT 기반 및 c-MAS 기반 최종 통합 결과

**조건:** NTU60 `xsub_val`, MotionBERT Backbone 고정, MLP 메타 분류기 학습(GT 40K) 및 평가

#### GT 데이터 기반 (16,487 val samples)

| Combo | 뷰 구성 | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | 0° (Baseline) | 50.77% | 68.84% | 68.54% | 73.47% | 73.23% |
| C2 | 0°, ±30° | 51.83% | 71.04% | 71.64% | 75.43% | 75.93% |
| C3 | 0°, ±45° | 51.87% | 71.81% | 72.60% | 76.12% | 76.44% |
| C4 | 0°, ±90° | 50.98% | 71.72% | 73.45% | 76.18% | 77.27% |
| C8 | All 7 Views | 51.90% | 72.52% | 73.87% | 76.82% | **77.84%** |

#### c-MAS 데이터 기반 (480 val samples, GT-trained models, zero-shot)

| Combo | 뷰 구성 | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | 0° (Baseline) | 40.21% | 57.92% | 56.46% | 59.58% | 60.21% |
| C2 | 0°, ±30° | 39.79% | 57.50% | 56.67% | 60.42% | 56.88% |
| C3 | 0°, ±45° | 40.00% | 56.67% | 53.96% | 59.38% | 55.42% |
| C4 | 0°, ±90° | 37.71% | 50.42% | 52.71% | 50.00% | 51.04% |
| C8 | All 7 Views | 40.21% | 53.75% | 53.33% | 55.83% | 53.96% |

> Raw 0° 정확도 갭: GT 50.77% vs c-MAS 40.21% (-10.56%p)

---

### 추가 실험 1: GT-trained MLP + c-MAS Views (Zero-shot Inference)

**조건:** MLP는 완벽한 GT 분포만 학습, 추론 시점에만 c-MAS 뷰 결합

| Combo | 뷰 구성 | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | GT 0° (Baseline) | 49.38% | 70.21% | 70.21% | **74.79%** | 73.75% |
| C2 | GT 0° + c-MAS ±30° | 45.21% | 67.50% | 56.67% | 74.38% | 62.71% |
| C3 | GT 0° + c-MAS ±45° | 45.00% | 66.67% | 58.75% | 72.08% | 63.75% |
| C4 | GT 0° + c-MAS ±90° | 43.12% | 57.50% | 54.58% | 62.50% | 62.92% |
| C8 | GT 0° + c-MAS All | 41.88% | 58.13% | 53.96% | 60.62% | 58.33% |

> c-MAS 뷰를 추가할수록 성능 하락 → GT-trained MLP가 c-MAS의 노이즈 분포에 미적응

---

### 추가 실험 2: GT+c-MAS 혼합 학습 (Mixed Training)

**조건:** MLP 학습 단계부터 c-MAS 데이터를 함께 노출, 노이즈 적응 유도 (소규모 샘플링 조건)

| Combo | 뷰 구성 | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|
| C1 | GT 0° (Baseline) | 44.17% | 43.12% | 68.12% | 69.17% |
| C2 | GT 0° + c-MAS ±30° | 28.33% | 33.75% | 64.58% | 60.62% |
| C3 | GT 0° + c-MAS ±45° | 28.12% | 31.46% | 62.29% | 55.83% |
| C4 | GT 0° + c-MAS ±90° | 22.29% | 26.88% | 58.13% | 55.21% |
| C8 | GT 0° + c-MAS All | 14.79% | 32.71% | 50.62% | 39.58% |

> 소규모 혼합 학습은 오히려 성능 저하 → 학습 데이터 규모 부족 + 도메인 불일치 복합 영향

---

## 주요 발견

1. **Feature-level fusion > Logit-level fusion**: 고차원 특징(2048-dim)을 융합하는 것이 저차원 로짓(60-dim) 융합보다 일관되게 우수
2. **GT 멀티뷰의 유효성 확인**: C8 Feat-Concat 77.84%로 단일뷰 대비 +4.61%p 향상
3. **c-MAS 도메인 갭 존재**: Raw 정확도 기준 -10.56%p 갭, GT-trained 모델의 zero-shot 적용 시 성능 회복 제한적
4. **c-MAS 뷰 추가 시 성능 하락**: 현재 파이프라인의 누적 오차(Kinect→NBA→COCO 이중 변환)가 품질 저하의 주원인

---

## 프로젝트 구조

```
final_project/
├── c-MAS/                          # 생성형 멀티뷰 파이프라인
│   ├── pipeline.py                 # 멀티뷰 생성 파이프라인 (--split val/train, --stage 1/2/3/all)
│   ├── eval.py                     # c-MAS 추론 평가 (--mode full/quick/time)
│   ├── model/                      # MDM 디퓨전 모델
│   ├── diffusion/                  # Gaussian diffusion 구현
│   ├── data_loaders/nba/           # NBA-16 포맷 데이터 로더
│   ├── sample/                     # MAS 샘플링 (mas.py, sampler.py)
│   ├── utils/                      # 공통 유틸리티
│   ├── save/yoga_diffusion_model/  # c-MAS pretrained 체크포인트 (200K steps)
│   ├── pipeline_out/               # Val 생성 결과 (npy, pkls, vis)
│   └── pipeline_train_out/         # Train 생성 결과 (npy, pkls)
│
└── MotionBERT/                     # 행동 인식 백본 + 융합 분류기
    ├── train.py                    # MLP 학습 (--mode logit/feat/mixed)
    ├── eval.py                     # 평가 (--mode gt_single/gt_combo/cmas/gt0_cmas)
    ├── make_table.py               # 결과 테이블 (--mode cmas/final)
    ├── proj_nba2coco.py            # NBA→COCO 좌표 변환 전처리
    ├── visualize_projection.py     # GT 골격 2D 투영 시각화
    ├── lib/                        # MotionBERT 모델/데이터/유틸
    ├── configs/action/             # 학습 설정 YAML
    ├── data/action/                # 전처리된 pkl 데이터
    │   ├── ntu60_gt_*.pkl          # GT 투영 데이터 (symlink, ~1.1GB each)
    │   └── ntu60_cmas_*.pkl        # c-MAS 생성 데이터 (~8MB each)
    ├── save/MB_ft_NTU60_xsub/      # MotionBERT pretrained 체크포인트
    └── view_attn_out/
        ├── checkpoints_logit_combo/  # Logit-level MLP 체크포인트
        ├── checkpoints_feat/         # Feature-level MLP 체크포인트
        ├── checkpoints_mixed/        # Mixed GT+c-MAS MLP 체크포인트
        ├── features/                 # 추출된 2048-dim 특징 (symlink, 3.1GB)
        ├── logits/                   # 추출된 60-dim 로짓
        └── vis/                      # 결과 시각화 이미지
```

---

## 실행 방법

### 환경

- c-MAS 실행: `conda activate cMAS`
- MotionBERT 실행: `conda activate POSCO`

### 1. c-MAS 가상 멀티뷰 생성

```bash
cd c-MAS

# Val subset 전체 파이프라인 (추론 → 시각화 → pkl 빌드)
python pipeline.py --split val --n_samples 500 --diffusion_steps 20 --stage all

# Train subset (추론 → pkl 빌드, 시각화 없음)
python pipeline.py --split train --per_class 8 --diffusion_steps 20 --stage 13

# 스테이지 개별 실행
python pipeline.py --split val --stage 1   # 추론만
python pipeline.py --split val --stage 2   # 시각화만
python pipeline.py --split val --stage 3   # pkl 빌드만
```

### 2. c-MAS 출력 평가

```bash
cd c-MAS

# MotionBERT까지 end-to-end 평가
python eval.py --mode full --n_eval 200

# 소수 샘플 시각화 테스트
python eval.py --mode quick --num_samples 3

# 추론 속도 벤치마크
python eval.py --mode time
```

### 3. MLP 융합 분류기 학습

```bash
cd MotionBERT

python train.py --mode logit   # Logit-level MLP (실험2)
python train.py --mode feat    # Feature-level MLP (실험2)
python train.py --mode mixed   # Mixed GT+c-MAS MLP (추가실험2)
```

### 4. 평가

```bash
cd MotionBERT

python eval.py --mode gt_single              # GT 단일뷰 베이스라인
python eval.py --mode gt_combo               # GT 멀티뷰 콤보 (실험2)
python eval.py --mode cmas                   # c-MAS zero-shot 전체 (실험2)
python eval.py --mode gt0_cmas               # GT 0° + c-MAS 뷰 (추가실험1)
```

### 5. 결과 테이블 출력

```bash
cd MotionBERT

python make_table.py --mode cmas             # c-MAS 결과 비교표
python make_table.py --mode final            # GT vs c-MAS 최종 비교표
python make_table.py --mode final --no_plot  # 텍스트만 출력
```

---

## 데이터 및 모델

| 항목 | 경로 | 비고 |
|---|---|---|
| NTU60 3D 원본 | `/workspace/clone_model/data/ntu60_3danno.pkl` | 공용 |
| GT 투영 데이터 (val, ×8 angles) | `MotionBERT/data/action/ntu60_gt_*.pkl` | symlink (~1.1GB each) |
| c-MAS 생성 데이터 (val, ×7 angles) | `MotionBERT/data/action/ntu60_cmas_*.pkl` | 직접 포함 (~8MB each) |
| c-MAS 생성 데이터 (train, ×7 angles) | `MotionBERT/data/action/ntu60_cmas_train_*.pkl` | 직접 포함 (~8MB each) |
| c-MAS pretrained | `c-MAS/save/yoga_diffusion_model/checkpoint_200000.pth` | 직접 포함 (209MB) |
| MotionBERT pretrained | `MotionBERT/save/MB_ft_NTU60_xsub/best_epoch.bin` | 직접 포함 (231MB) |
| 추출된 GT features | `MotionBERT/view_attn_out/features/` | symlink (3.1GB) |
| Logit-level MLP 체크포인트 | `MotionBERT/view_attn_out/checkpoints_logit_combo/` | 직접 포함 |
| Feature-level MLP 체크포인트 | `MotionBERT/view_attn_out/checkpoints_feat/` | 직접 포함 |
| Mixed MLP 체크포인트 | `MotionBERT/view_attn_out/checkpoints_mixed/` | 직접 포함 |