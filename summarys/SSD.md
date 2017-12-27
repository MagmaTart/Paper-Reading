# SSD (Single Shot Detector)
![](http://postfiles8.naver.net/MjAxNzA1MTdfMTkx/MDAxNDk0OTk2NzkwODUw.Y1B8MR2m9YAG_UoAQmZPMKFepv_U6YzY4PIrwe70dKsg.otkjEG53NYfCgWXKQWWRPsjIp3rQhOdNTLq4AQyOILog.PNG.sogangori/fig1.PNG?type=w1)

##### 네트워크 구조
![](http://postfiles13.naver.net/MjAxNzA1MTdfNTYg/MDAxNDk0OTkzMTM4OTY0.HgEdZuksjx5KRU-bLBoMzO5m7gGtvKoT4zwKvGb5VLEg.aXdVrKip7SzTI-Mq9hCfO8FUXtHQRsjgMj1v1Dkl60og.PNG.sogangori/architectureSSD.PNG?type=w1)

SSD는, YOLO와 같이 통합된 하나의 네트워크로 Object Detection을 수행하는 네트워크입니다. SSD의 가장 큰 특징은, 크기가 다른 여러 개의 Feature Map을 사용하여 각각 다른 Detection을 수행한다는 것입니다. 각 Feature Map을 일정한 수의 Grid Cell로 나누고, 각 Feature Map마다 일정한 종횡비(aspect ratio)를 가지는 Default box들을 생성합니다. 여기서 default box는 Bounding box와 같은 역할을 수행합니다. 모든 Default box들에 대해서 신뢰도 점수 즉, 물체가 안에 포함되어 있을 확률을 계산하고, 일정 Threshold 이하라면 그 box를 버립니다. 그리고 학습을 수행하는데, 기본적으로 정답 box와의 Jaccard Overlap Score, 즉 IoU가 0.5 이상인 모든 box들이 살아남아 학습을 진행합니다. 

Bounding Box의 학습은 신뢰도 점수가 낮은 negative box들을 우선적으로 학습시키는 방향으로 진행합니다. 이렇게 되면 전체 평균 Loss가 감소하는 효과를 볼 수 있기 때문입니다.
