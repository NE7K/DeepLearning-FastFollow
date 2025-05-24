# DeepLearning-FastFollow

다양한 딥러닝 프로젝트를 실습하며 학습한 결과물들을 모은 저장소입니다. 주요 프로젝트는 사망 확률 예측, 개/고양이 분류, 사람 얼굴 그리기, 쇼핑 이미지 분류 등으로 구성되어 있습니다. 

## 📂 Project Structure
```
DeepLearning-FastFollow/
│
├── Checkpoint/                         # 모델 체크포인트 디렉토리
│   ├── checkpoint                      # 체크포인트 메타 정보
│   ├── mnist.data-00000-of-...        # MNIST 모델 데이터 파일
│   ├── mnist.index                    # MNIST 모델 인덱스
│
├── CommentAi/                         # 댓글 AI 관련 코드
│
├── Composition/                       # 구성 관련 기능 (미정의)
│
├── Data/                              # 데이터 파일 모음
│
├── dataset/                           # 데이터셋 1
├── dataset2/                          # 데이터셋 2
│
├── DeadProbability/                   # 사망 확률 예측 프로젝트
│   ├── Data/                          # 데이터 디렉토리
│   ├── SaveModel/                     # 학습된 모델 저장소
│   ├── DeadProbability_Pre...py      # 사망 확률 전처리 코드
│   ├── DeadProbability.py            # 사망 확률 메인 코드
│
├── DogCatAnalysis/                    # 개/고양이 분류 프로젝트
│   ├── DogCatAnalysis.py             # 메인 분석 코드
│
├── DrawHumanFace/                     # 사람 얼굴 그리기 프로젝트
│   ├── img_align_celeba/             # CelebA 이미지 데이터셋
│   ├── trainingimg/                  # 학습용 이미지
│   ├── DrawHumanAi.py                # 메인 모델 코드
│
├── LogFile/                           # 학습 로그 저장소
│   ├── Log_Model_1747547...          # 로그 파일
│
├── ModelImage/                        # 모델 이미지 저장 디렉토리
│   ├── Model_FunctionalAPI...        # Functional API 모델
│
├── SaveModel/                         # 모델 저장소 (다중 프로젝트 포함)
│   ├── DogCatAnalysis1/              # DogCat 모델 저장
│   ├── SaveModel/                    # 기타 저장 경로
│   ├── ShopImageTraining/            # 쇼핑 이미지 모델
│   ├── CompositionLSTM.keras         # LSTM 모델
│
├── ShopImage/                         # 쇼핑 이미지 분류 프로젝트
│   ├── ShopImageTraining_...py       # 학습 코드들
│   ├── ShopImageTraining.py          # 메인 학습 코드
│
├── train/                             # 훈련 관련 파일 (미정의)
│
├── .gitignore                         # Git 무시 설정
├── GetModel.py                        # 모델 가져오기 유틸
├── GpuTest.py                         # GPU 테스트 코드
├── inception_v3.h5                    # 사전 학습된 Inception 모델
├── README.md                          # 프로젝트 설명 파일
├── sampleSubmission.csv              # 샘플 제출 양식
├── TransferLearning.py               # 전이 학습 예제
```

## Comment Ai
loss: 0.0641 - accuracy: 0.9775 - val_loss: 0.0544 - val_accuracy: 0.9812

## Composition

## DeadProbability

## DogCatAnalysis

## DrawHumanFace
epochs : 46, 최종 loss는 Discriminator : 0.5758713781833649, GAN : 2.515842914581299

실제 트레이닝된 이미지 :

![9](https://github.com/user-attachments/assets/ffa472f0-87be-47eb-9b1e-8974bd4d8945)
![8](https://github.com/user-attachments/assets/f5e4e296-e598-4b9d-8f1f-edd76d2b1e8f)
![7](https://github.com/user-attachments/assets/bf875bed-8143-444f-bdac-d28e4bc919b8)
![6](https://github.com/user-attachments/assets/7417314c-b7b0-4a8d-8c92-8bbbd186e8d3)
![5](https://github.com/user-attachments/assets/3d56072c-936d-4de5-877f-e6d2e516f301)
![4](https://github.com/user-attachments/assets/1fda903e-304d-468f-932f-1c5c156b08db)
![3](https://github.com/user-attachments/assets/59bbb0c9-35ab-49d8-b61a-926ecddd5d32)
![2](https://github.com/user-attachments/assets/8bafb3f6-3bda-4ef5-bd17-b2e6b0e9ab57)
![1](https://github.com/user-attachments/assets/92383746-9b20-426a-8ea7-2289392f8cb1)
![0](https://github.com/user-attachments/assets/eeae3746-8ed2-40b3-8364-ae744e253820)

## 🧪 Utill
GpuTest.py: GPU가 정상적으로 작동하는지 확인

GetModel.py: 저장된 모델 로드 및 테스트
