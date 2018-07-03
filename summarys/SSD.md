# SSD: Single shot MultiBox Detector
### Abstract
이 논문에서는, 단일 NN을 사용해서 이미지 내 물체를 검출하는 방법을 제시합니다. SSD는 피처 맵 내에서 바운딩 박스의 출력 공간을 다양한 종횡비와 크기를 가진 Default 박스들로 나눕니다. 네트워크는 각 Default box 내에 있는 물체의 클래스를 예측하고, 물체의 모양에 맞는 Box를 예측합니다. 또한 네트워크는 다양한 크기를 가진 여러 Feature map에서의 예측들을 결합해 사용하는데, 이는 다양한 사이즈의 물체를 잡아내기 위해서입니다. SSD는 Object Proposal을 필요로 하는 Detection 방법들과는 달리, Proposal을 생성하는 과정과, 픽셀과 특징을 다시 Resampling하는 과정을 완전히 없애 버리고, 하나의 네트워크에 모든 연산을 묶어놓았습니다. 이는 SSD를 쉽게 트레이닝할 수 있게 하고, Detection을 필요로 하는 시스템에 쉽게 이식할 수 있게 합니다. PASCAL VOC, COCO, ILSVRC 데이터셋을 이용한 실험에서 SSD는 추가적인 Object Proposal을 생성하는 방법들과 비교했을때도 경쟁력있는 정확도를 달성했습니다. 게다가 트레이닝과 테스트 때 연합된 네트워크를 사용하기 때문에 좀 더 빠릅니다. VOC2007 데이터셋 내 300 x 300 크기의 이미지 입력에 대해서 NVIDIA Titan X로 학습한 SSD는 74.3%의 mAP와 59 FPS의 성능을 보였고, 512 x 512 이미지에 대해서 76.9%의 mAP를 보여 Faster R-CNN 모델보다 더 좋은 성능을 보였습니다. 다른 Single-Stage Detection 방법들과 비교해도 SSD는 작은 input에도 더 높은 정확도를 보였습니다.

### 1. Introduction
현재의 State-of-the-art Object Detection 시스템은 대부분 물체의 경계 박스(Bounding box)를 찾고, 각 박스마다 Feature를 추출한 다음, 잘 훈련된 분류기를 가져다가 분류하는 식으로 구성되어 있습니다. 이와 같은 방법이 처음으로 제시된 [Selective-Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)부터 최근의 [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)까지, 이 방식의 Detection 모델들은 좋은 성능을 보여줬습니다. 하지만 이런 모델들은 물체 검출에 보이는 정확도에 비해, 연산량 면에서 너무 무거워 고성능의 하드웨어가 필요했고, 임베디드 시스템이나 실시간 시스템에 올리기에는 무리가 있었습니다.
현재까지 가장 빠른 Detection 시스템인 Faster R-CNN이 초당 7 프레임의 속도를 보여주고 있습니다. 수많은 연구자들이 더 빠른 모델을 만들기 위해 실험했지만, 속도가 빨라질수록 정확도는 낮아지는 결과를 보이고 있었습니다.

이 논문에서는 최초로 픽셀이나 Feature 단위의 Resample을 하지 않고 경계 박스를 찾는 과정이 없으면서도 이런 과정을 거치는 네트워크들과 동일하거나 더 좋은 성능을 보이는 모델을 제안합니다. 결과는 성능과 속도 면에서 월등했습니다. (VOC2007 데이터셋 기반 mAP 74.3% 59 FPS)
속도 향상의 주 요인은 경계 박스를 찾는 과정과 Feature의 Resampling 과정을 없앤 것이었습니다. 논문에서 제안하는 모델은 물체의 카테고리와 경계 박스 위치의 오프셋을 예측하기 위해 작은 컨볼루션 필터를 사용합니다. 또 이런 과정은 다양한 크기를 가진 물체들의 Detection을 위해, 네트워크 내 여러 크기의 Feature Map에서 동시에 적용됩니다. 이 방법을 사용해서, 상대적으로 낮은 해상도의 Input 이미지를 사용해도 빠른 속도의 Detection을 할 수 있었습니다.

### 2.1 Model 
SSD는 __Non-Maximum Suppression__ 을 통해 최종적으로 남은 고정된 크기의 경계 박스들과 그 박스 안에 존재하는 물체를 예측한 스코어를 생성하는 Convolutional Network를 기반으로 작동합니다. 논문에서 __Base network__ 라고 부르는 네트워크의 초반 레이어들은, 일반적인 고성능 이미지 분류 네트워크와 같은 구조를 사용합니다. 거기에다 추가적인 구조를 더해 SSD approach를 완성합니다.

#### Multi-scale feature map
Base network의 끝에 Convolutional Feature layer들을 추가했습니다. 이 레이어들은 사이즈가 점점 줄어드는데, 이는 다양한 스케일의 Detection을 가능하게 합니다. 각 레이어마다 Detection을 위한 Convolutional Model이 다르다고 이야기하고 있는데, 즉 각 레이어마다의 Detection이 다르다는 것입니다. 이를 추가한 모델의 모양은 아래 __Figure 2__ 와 같습니다.

![Figure 2](../images/SSD/figure2.PNG)

__Figure 2__

#### Convolutional Predictors
추가된 각 Feature layer들은 각각의 Convolutional filter들을 사용해 고정된 양의 Detection 예측을 만들 수 있습니다. 이는 __Figure 2__ 에서 SSD 구조의 상단에 표현되어 있습니다. m x n의 사이즈와 p 채널을 가진 Feature layer에서, 3 x 3 x p의 작은 커널은 물체 카테고리의 예측 점수와 Default box의 위치와 관련 있는 모양 offset을 만들어냅니다. 커널이 적용되는 m x n 크기 내 모든 위치에서 이 출력이 만들어집니다.
경계 박스 offset 출력 값은 Feature Map에서 Default box가 있는 각 부분들에 대해 모두 측정됩니다. 이는 YOLO에서, Grid로 나누어진 Feature Map의 모든 부분에 대해 경계 박스 Prediction 출력이 각각 만들어지는 것과 동일합니다. 대신 YOLO는 중간에 Fully-Connected Layer를 사용하고, SSD는 대신 Convolutional filter를 사용한다는 점이 다릅니다.

#### Default boxes and aspect ratios
네트워크의 상단의 여러 Feature map 상에서, 각각의 Feature map cell에 대해 Default 경계 박스 세트를 설정합니다. Feature map cell은 Feature map 상의 하나의 픽셀이라고 보면 됩니다. Default 박스들은 Convolution과 같은 방법으로 Feature map을 돌면서, 각 cell에 대해 박스의 위치를 고정합니다. 그 후에, 각 cell에서 연결된 Default box의 모양을 나타내는 Shape offset과 box 내 어떤 클래스의 물체가 존재하는지를 의미하는 클래스별 점수를 예측합니다.

주어진 위치에서의 ![](https://latex.codecogs.com/gif.latex?k)개의 box가 주어지면, 각 박스에 대해 클래스 ![](https://latex.codecogs.com/gif.latex?c)개에 대한 스코어와 Box shape와 관련된 4개의 offset을 예측합니다. 그러므로 각 cell에서 ![](https://latex.codecogs.com/gif.latex?%28c&plus;4%29k)개의 filter가 사용되고, 출력도 같은 개수입니다. 따라서 ![](https://latex.codecogs.com/gif.latex?m%20%5Ctimes%20n) 크기의 feature map 상의 출력은 ![](https://latex.codecogs.com/gif.latex?%28c&plus;4%29kmn)개입니다. SSD에서의 Default box는 Faster R-CNN에서의 __anchor box__ 와 비슷하지만, 다양한 해상도에서의 다양한 Feature map을 사용한다는 점이 다릅니다. 다양한 Feature map에서 각각 다른 default box를 가져가는 것은 가능한 Output box shape를 효과적으로 분리하도록 만들어줍니다.

### 2.2 Training
모델을 학습하는 과정에서 SSD가 Region proposal을 사용하는 대부분의 Detector 모델들과 가지는 차이는, 특정 Output을 처리하기 위해 Ground truth가 필요하다는 것입니다. 이 중 일부 방법은 YOLO나 Faster R-CNN에서 Region proposal을 만드는 단계에서도 사용됩니다. 이 방법을 사용하면, Loss와 역전파 과정이 네트워크 학습에 End-to-End로 적용될 수 있습니다. 또한 트레이닝에는 Hard Negative mining과 Data augmentation 기법들이 사용됩니다.

#### Matching strategy
트레이닝 과정에서, Ground truth를 예측한 Default box를 결정하고 네트워크의 학습에 반영하는 것이 필요합니다. 각 Ground truth box에 대해서 다양한 위치, 종횡비, 스케일에 따라 여러 Default box들을 선택합니다. 그리고 그 Default box들 중 Ground truth와의 [Jaccard Overlap](https://blog.naver.com/leesoo9297/221159046121) 이 가장 큰 Box를 찾아내기 시작합니다. 이 과정에서 Jaccard Overlap 점수가 0.5 이하인 Default box는 지웁니다. 이런 과정은 네트워크의 학습 문제를 단순화합니다. 네트워크가 최고로 Overlap이 가장 큰 박스를 선택하도록 하는 것이 아니라, 위와 같은 방식으로 Overlap이 큰 여러 박스가 존재할 경우에 높은 점수를 주는 방향으로 트레이닝할 수 있습니다.

#### Training Objective
SSD의 트레이닝 목적 함수는 MultiBox의 목적 함수를 여러 Object category를 핸들링할 수 있게 확장한 버전입니다. ![](https://latex.codecogs.com/gif.latex?x%5Ep_%7Bij%7D%20%3D%20%5C%7B1%2C0%5C%7D)는 ![](https://latex.codecogs.com/gif.latex?i)번째 Default box가 클래스 ![](https://latex.codecogs.com/gif.latex?p)의 ![](https://latex.codecogs.com/gif.latex?j)번째 Ground truth box를 가리키는 박스인지를 나타내는 Indicator입니다. 이 값이 1이면 참이고, 0이면 거짓입니다. 이를 사용한 SSD의 Loss는 아래와 같이, Localization Loss와 Confidence Loss의 가중합으로 구성되어 있습니다.

![](https://latex.codecogs.com/gif.latex?L%28x%2Cc%2Cl%2Cg%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%28L_%7B%5Ctext%7Bconf%7D%7D%28x%2Cc%29%20&plus;%20%5Calpha%20L_%7B%5Ctext%7Bloc%7D%7D%28x%2Cl%2Cg%29%29)

이제 각 Loss를 자세히 살펴보겠습니다. 그 전에, 아래와 같이 정의하고 넘어가겠습니다.

![](https://latex.codecogs.com/gif.latex?N) : 매칭된 Default box의 개수

![](https://latex.codecogs.com/gif.latex?l) : 예측한 Box

![](https://latex.codecogs.com/gif.latex?g) : Ground truth box

![](https://latex.codecogs.com/gif.latex?d) : Default box

![](https://latex.codecogs.com/gif.latex?cx%2C%20cy) : Box의 Center 위치

![](https://latex.codecogs.com/gif.latex?w%2C%20h) : Box의 WIdth와 Height

예측한 Box를 Default box의 Offset에 맞추어 회귀 학습합니다. 그 과정에서 Smooth L1 Loss가 사용됩니다. 회귀 작업을 수행하는 Localization Loss는 아래와 같이 정의됩니다.

![](https://latex.codecogs.com/gif.latex?L_%7B%5Ctext%7Bloc%7D%7D%28x%2Cl%2Cg%29%20%3D%20%5Csum%5E%7BN%7D_%7Bi%20%5Cin%20Pos%7D%20%5C%20%5Csum_%7Bm%20%5Cin%20%5C%7Bcx%2Ccy%2Cw%2Ch%5C%7D%7D%20x%5Ek_%7Bij%7D%5Ctext%7Bsmooth%7D_%7B%5Ctext%7BL1%7D%7D%28l%5Em_i%20-%20%5Chat%7Bg%7D%5Em_i%29)

![](https://latex.codecogs.com/gif.latex?%5Chat%7Bg%7D%5E%7Bcx%7D_j%20%3D%20%28g%5E%7Bcx%7D_j%20-%20d%5E%7Bcx%7D_i%29%20/%20d%5Ew_i%2C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5Chat%7Bg%7D%5E%7Bcy%7D_j%20%3D%20%28g%5E%7Bcy%7D_j%20-%20d%5E%7Bcy%7D_i%29%20/%20d%5Eh_i)

![](https://latex.codecogs.com/gif.latex?%5Chat%7Bg%7D%5Ew_j%20%3D%20%5Clog%20%5CBig%28%5Cfrac%7Bg%5Ew_j%7D%7Bd%5Ew_i%7D%5CBig%29%2C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5Chat%7Bg%7D%5Eh_j%20%3D%20%5Clog%20%5CBig%28%5Cfrac%7Bg%5Eh_j%7D%7Bd%5Eh_i%7D%5CBig%29)

이 Loss에서, ![](https://latex.codecogs.com/gif.latex?x%5Ek_%7Bij%7D%20%3D%200)일 경우에는 Loss가 0이 됩니다. 이는 Ground truth와 매칭되는 박스인 경우에만 Loss에 반영하겠다는 의미를 담고 있습니다.

Confidence Loss는 아래와 같이 정의됩니다.

![](https://latex.codecogs.com/gif.latex?L_%7B%5Ctext%7Bconf%7D%7D%28x%2Cc%29%20%3D%20-%5Csum%5EN_%7Bi%20%5Cin%20Pos%7D%20x%5Ep_%7Bij%7D%5Clog%28%5Chat%7Bc%7D%5Ep_i%29%20-%20%5Csum_%7Bi%20%5Cin%20Neg%7D%20%5Clog%28%5Chat%7Bc%7D%5E0_i%29%20%5C%20%5C%20%5C%20%5C%20%5C%20%5Ctext%7Bwhere%7D%20%5C%20%5C%20%5C%20%5Chat%7Bc%7D%5Ep_i%20%3D%20%5Cfrac%7B%5Cexp%28c%5Ep_i%29%7D%7B%5CSigma_p%20%5Cexp%28c%5Ep_i%29%7D)

이 Loss는, 매칭된 Box가 Negative면 클래스가 0으로 고정되고, Positive면 Indicator ![](https://latex.codecogs.com/gif.latex?x%5Ep_%7Bij%7D)의 값에 따라 Loss 반영이 결정됩니다.

두 Loss를 가중합한 최종 Loss에서, 가중치 ![](https://latex.codecogs.com/gif.latex?%5Calpha)는 교차 검증을 통해 1로 설정했다고 합니다.

#### Choosing scales and aspect ratios for default boxes
다양한 물체의 크기를 다루기 위해서, 이미지를 다양한 사이즈로 처리한 후 결과를 합치는 여러 방법들이 제시되었습니다. 그러나, 단일 네트워크 상에서 여러 다양한 네트워크의 Feature map을 사용하여 Prediction하면, 모든 물체 크기에 대해서 Parameter를 공유하면서도 위와 같은 효과를 낼 수 있습니다. 이전의 연구 결과들은 하위 레이어의 Feature map을 사용하면 물체의 디테일한 특징을 더 잘 잡을 수 있어 Semantic segmentation의 품질을 높여주는 결과를 가져옴을 보여주었습니다. 비슷하게, ParseNet에서는 Feature map에서의 Global Context Pool이 Segmentation의 결과를 smooth하게 만들어준다는 것을 보였습니다.

![Figure 1](../images/SSD/figure1.PNG)

__Figure 1__


이런 방법들에 영향을 받아, SSD에는 Detection에 하위와 상위 Feature map 두 개를 사용합니다. 위에 있는 __Figure 1__ 은 두 개의 예시 8 x 8과 4 x 4 Feature map을 보여줍니다. 실제로는 적은 연산량 증가만 가지고서도 많은 Feature map을 더 사용할 수 있습니다.
네트워크의 다른 층에서 여러 Feature map을 사용하는 것은, 다양한 Receptive field의 크기를 가져가기 위함으로 해석할 수 있습니다. 논문에서는, 다행스럽게도 SSD 모델 구조 내에서 Default box가 각 Feature map 레이어의 실제 Receptive Field와 같을 필요는 없다고 말합니다. 논문에서는 특정 Feature map이 특정한 물체의 스케일에 반응하도록 Default box를 움직이게 학습했습니다. ![](https://latex.codecogs.com/gif.latex?m)개의 Feature map을 Prediction을 위해 사용한다고 할 때, 각 Feature map에 대한 Default box들의 스케일은 다음과 같이 계산됩니다.

![](https://latex.codecogs.com/gif.latex?s_k%20%3D%20s_%7B%5Ctext%7Bmin%7D%7D%20&plus;%20%5Cfrac%7Bs_%7B%5Ctext%7Bmax%7D%7D%20-%20s_%7B%5Ctext%7Bmin%7D%7D%7D%7Bm%20-%201%7D%28k%20-%201%29%2C%20%5C%20%5C%20%5C%20%5C%20%5C%20k%20%5Cin%20%5B1%2Cm%5D)

![](https://latex.codecogs.com/gif.latex?k)는 모든 Feature map을 순회하는 변수라고 보면 되며, Default box의 Scale은 위에서 4x4, 8x8과 같이 이야기한 Default box의 크기를 말합니다. 식에 따르면, 상위 레이어에서는 큰 Default box, 하위 레이어에서는 작은 Default box를 가지도록 유도됩니다.

예를 들어 ![](https://latex.codecogs.com/gif.latex?s_%7B%5Ctext%7Bmin%7D%7D)이 0.2이고 ![](https://latex.codecogs.com/gif.latex?s_%7B%5Ctext%7Bmax%7D%7D)가 0.9일 떄는, 가장 하위 레이어의 Default box는 0.2배의 Scale을 가지고 가장 상위 레이어의 Default box는 0.9의 Scale을 가지며, 그 사이는 일정하게 크기가 변화한다는 이야기입니다. 또 논문에서는 각 Default box마다 여러 Aspect ratio(종횡비)를 또 적용하였는데, ![](https://latex.codecogs.com/gif.latex?a_r%20%5Cin%20%5C%7B1%2C%202%2C%203%2C%20%5Ctfrac%7B1%7D%7B2%7D%2C%20%5Ctfrac%7B1%7D%7B3%7D%20%5C%7D)와 같이 나타내어집니다. 종횡비가 적용된 각 Default box의 가로와 세로 크기는 다음과 같이 계산됩니다.

![](https://latex.codecogs.com/gif.latex?w%5Ea_k%20%3D%20s_k%20%5Csqrt%7Ba_r%7D%2C%20%5C%20%5C%20%5C%20%5C%20%5C%20h%5Ea_k%20%3D%20s_k%20/%20%5Csqrt%7Ba_r%7D)

이렇게 나온 5개의 Box에, 사이즈 ![](https://latex.codecogs.com/gif.latex?s%5E%5Cprime_k%20%3D%20%5Csqrt%7Bs_ks_%7Bk&plus;1%7D%7D)와 종횡비 1을 가지는 Default box를 하나 추가해서 총 6개의 Default box가 Feature map의 각 Location마다 생성됩니다. 또한 각 Default box의 Center 위치를 ![](https://latex.codecogs.com/gif.latex?%28%5Ctfrac%7Bi&plus;0.5%7D%7B%7Cf_k%7C%7D%2C%20%5Ctfrac%7Bj&plus;0.5%7D%7B%7Cf_k%7C%7D%29)로 설정합니다. 여기서 ![](https://latex.codecogs.com/gif.latex?%7Cf_k%7C)는, ![](https://latex.codecogs.com/gif.latex?i%2C%20j%20%5Cin%20%5B0%2C%20%7Cf_k%7C%29)를 만족시키는 ![](https://latex.codecogs.com/gif.latex?k)번째 정사각형 Feature map의 한 변 길이를 의미합니다. 실제로 이런 방법은, 특정 ㄷ이터셋에 대해 Default box들의 분포가 딱 알맞게 이루어지도록 디자인할 수 있습니다.

많은 Feature map들의 모든 위치에서 다양한 종횡비와 스케일을 가진 Default box 예측을 합치는 작업을 통해, 다양한 물체의 사이즈와 모양을 커버할 수 있는 넓은 Prediction 세트를 얻어낼 수 있습니다. __Figure 1__ 을 보면, 크기가 큰 강아지는 Default box가 더 넓은 영역을 커버하는 4x4 크기 Feature map에서 찾아집니다. 하지만 8x8 크기 Feature map에서는 강아지의 크기와 모양에 맞는 어떤 Default box도 찾을 수 없습니다. 따라서 8x8 크기의 Feature map에서는 강아지의 일부가 들어있는 박스들이 다 Negative로 인식됩니다. 이런 방식의 Detection 진행이 다양한 크기와 모양을 가진 여러 물체들을 효과적으로 인식할 수 있게 합니다.

#### Hard negative mining
위에서 설명한 Matching 과정을 거친 후에, 수많은 Default box가 Negative로 판단됩니다. 이는 Positive와 Negative Sample 갯수 간에 큰 불균형을 가져옵니다. 따라서 논문에서는 모든 Negative Sample들을 사용하지 않고, Confidence loss가 높은 Negative box 순으로 뽑아서 Positive와 Negative의 비율이 최대 3:1이 되도록 맞춥니다. 이 과정을 통해 최적화가 빨라지고 안정적인 트레이닝을 가능하게 할 수 있었습니다.

#### Data augmentation
모델이 다양한 물체의 크기와 모양에 강건(robust)해지도록, 각각의 트레이닝 이미지는 다음과 같은 방법 중 하나를 선택해 샘플링됩니다.
- 입력 이미지 전체를 사용
- 물체와의 최소 jaccard overlap이 0.1, 0.3, 0.5, 0.7, 0.9인 patch를 추출해서 사용
- 랜덤하게 patch를 추출해서 사용

샘플링된 각 patch의 크기는 원본 이미지 사이즈의 ![](https://latex.codecogs.com/gif.latex?0.1%20%5Csim%201)배 사이이고, 종횡비는 원본 이미지의 ![](https://latex.codecogs.com/gif.latex?%5Ctfrac%7B1%7D%7B2%7D%20%5Csim%202)배 사이입니다. 또한, Ground Truth box의 center가 샘플링된 patch에 존재한다면 Overlap된 부분을 유지합니다. 이 샘플링 과정을 거치고 나면, 각 patch들은 고정된 크기로 resize되고, 0.5의 확률로 수평 반전(Horizontally flip)됩니다. 또한 여러 이미지의 왜곡도 같이 수행됩니다.