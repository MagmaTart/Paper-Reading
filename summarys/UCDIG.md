# UCDIG

### Introduction

Source domain ![](https://latex.codecogs.com/gif.latex?S)와 Target domain ![](https://latex.codecogs.com/gif.latex?T)에 대해서, 어떤 샘플 ![](https://latex.codecogs.com/gif.latex?x)에 대한 Generator Function ![](https://latex.codecogs.com/gif.latex?G%20%3A%20S%20%5Crightarrow%20T)가 정의될 때, ![](https://latex.codecogs.com/gif.latex?G%28x%29)는 ![](https://latex.codecogs.com/gif.latex?S)에서 뽑은 샘플을 ![](https://latex.codecogs.com/gif.latex?T) 도메인의 샘플으로 변환하는 작업을 한다. 이 논문의 목적은 이를 수행하는 GAN 네트워크를 만드는 것이다. ![](https://latex.codecogs.com/gif.latex?G%28x%29)는 ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20S)이든 ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20T)이든 상관없이 ![](https://latex.codecogs.com/gif.latex?T) 도메인의 이미지와 구별하지 못하는 샘플을 만들도록 학습된다. 그리고 이렇게 학습되는 네트워크를 __Domain Transfer Network(DTN)__ 이라고 한다.

Source domain ![](https://latex.codecogs.com/gif.latex?S)에서 어떤 분포 ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_S)에 따라 독립 항등 분포(i.i.d)로 샘플링된 레이블링 되지 않은 세트 ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bs%7D)가 주어지고, Target domain의 세트 ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bt%7D) 또한 ![](https://latex.codecogs.com/gif.latex?T)에서 같은 방법으로 ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_T)에 따라 주어진다. 함수 ![](https://latex.codecogs.com/gif.latex?f)가 주어지는데, 이 함수는 입력으로 어떤 도메인의 샘플이 들어오던지 도메인 ![](https://latex.codecogs.com/gif.latex?T)로 매핑하는 함수이다. 그리고 어떤 weight인 ![](https://latex.codecogs.com/gif.latex?%5Calpha)와 metric ![](https://latex.codecogs.com/gif.latex?d)가 같이 주어질 때,  DTN이 최소화하고자 하는 목적 함수는 다음과 같다.

![](https://latex.codecogs.com/gif.latex?R%20%3D%20R_%7BGAN%7D%20&plus;%20%5Calpha%20R_%7BCONST%7D)

![](https://latex.codecogs.com/gif.latex?R_%7BGAN%7D%20%3D%20%5Cunderset%7BD%7D%7Bmax%7D%20%5C%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_S%7D%20%5Clog%5B1%20-%20D%28G%28x%29%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_T%7D%20%5Clog%5BD%28x%29%5D)

![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_S%7D%20d%28f%28x%29%2C%20f%28G%28x%29%29%29)

![](https://latex.codecogs.com/gif.latex?R_%7BGAN%7D)은 GAN의 목적 함수를 Domain Transfer 문제에 맞게 변형시킨 것이다. Discriminator ![](https://latex.codecogs.com/gif.latex?D)는 입력이 Target domain의 샘플일 때 1을 분류하도록 학습시킨다. 따라서 앞 항에서는 Source domain ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_S)에서 샘플링한 ![](https://latex.codecogs.com/gif.latex?x)를 Discriminator에 넣었을 때, Target domain의 샘플로 인식해서 1로 분류하도록 유도하고 있다. 뒤 항은 ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_T)에서 샘플링된 Target domain의 샘플을 잘 분류하도록 만드는 항이다.

![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D)는 f-constancy를 보장하는 Term이다. 함수 ![](https://latex.codecogs.com/gif.latex?f)의 입력이 Source domain에서 나온 샘플일 떄와 Generator를 거쳤을 때, 두 경우의 차이를 최소화하려고 하고 있다. 논문에서는 metric ![](https://latex.codecogs.com/gif.latex?d)에 MSE를 사용하고 있다. 이 말은 곧, ![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D)가 작아질수록 함수 ![](https://latex.codecogs.com/gif.latex?f)에 대해 ![](https://latex.codecogs.com/gif.latex?x)와 ![](https://latex.codecogs.com/gif.latex?G%28x%29)가 같아진다는 말이 된다. 함수 ![](https://latex.codecogs.com/gif.latex?f)는 도메인 ![](https://latex.codecogs.com/gif.latex?T)에 대해서 정의되어있기 때문에, 이는 Source domain의 샘플과 Generator를 거친 샘플의 '의미', 즉 Context를 같다고 인식하고 있다는 뜻으로 해석할 수 있다.

### Loss function

DTN의 핵심은 다른 네트워크도 그렇듯이 Loss 함수다. 먼저 DTN에서는 Generator ![](https://latex.codecogs.com/gif.latex?G)를 ![](https://latex.codecogs.com/gif.latex?g%20%5Ccirc%20f)로 보고 있다. ![](https://latex.codecogs.com/gif.latex?f)는 입력된 이미지에서 Context를 encode하는 함수이고, ![](https://latex.codecogs.com/gif.latex?g)는 그 context를 바탕으로 다시 샘플을 Generate하는 함수이다.

DTN의 Loss 함수는 Discriminator에 대한 ![](https://latex.codecogs.com/gif.latex?L_D)와 Generator에 대한 ![](https://latex.codecogs.com/gif.latex?L_G) 둘로 이루어져 있다. 먼저 ![](https://latex.codecogs.com/gif.latex?L_D)는 다음과 같이 정의된다.

![](https://latex.codecogs.com/gif.latex?L_D%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20%5Clog%20D_1%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_2%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_3%28x%29)

![](https://latex.codecogs.com/gif.latex?D)는 삼중 분류를 수행하는 함수이고, ![](https://latex.codecogs.com/gif.latex?D_i%28x%29)는 입력 ![](https://latex.codecogs.com/gif.latex?x)가 클래스 ![](https://latex.codecogs.com/gif.latex?i)에 속할 확률을 반환한다. 따라서 Loss ![](https://latex.codecogs.com/gif.latex?L_D)는 

- 샘플 ![](https://latex.codecogs.com/gif.latex?x)로 생성한 이미지에서 ![](https://latex.codecogs.com/gif.latex?x)가 Source domain에서 나왔는지 (![](https://latex.codecogs.com/gif.latex?D_1))
- 샘플 ![](https://latex.codecogs.com/gif.latex?x)로 셍성한 이미지에서 ![](https://latex.codecogs.com/gif.latex?x)가 Target domain에서 나왔는지 (![](https://latex.codecogs.com/gif.latex?D_2))
- 샘플 ![](https://latex.codecogs.com/gif.latex?x)가 Target domain에서 나왔는지 (![](https://latex.codecogs.com/gif.latex?D_3))

이 셋으로 이루어져 있다고 볼 수 있다.

다음은 Generator의 Loss ![](https://latex.codecogs.com/gif.latex?L_G)이다. ![](https://latex.codecogs.com/gif.latex?L_G%20%3D%20L_%7BGANG%7D%20&plus;%20%5Calpha%20L_%7BCONST%7D%20&plus;%20%5Cbeta%20L_%7BTID%7D%20&plus;%20%5Cgamma%20L_%7BTV%7D)로 정의된다. ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%2C%20%5Cbeta%20%2C%20%5Cgamma)는 Weight이다.
먼저 ![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D)부터 살펴보자.

![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D%20%3D%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20%5Clog%20D_3%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_3%28g%28f%28x%29%29%29)

식의 의미는 간단하다. ![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D)을 최소화하려면, 샘플 ![](https://latex.codecogs.com/gif.latex?x)가 Source domain에서 샘플링됬던 Target domain에서 샘플링됬던지간에, ![](https://latex.codecogs.com/gif.latex?x)로부터 생성된 이미지 ![](https://latex.codecogs.com/gif.latex?g%28f%28x%29%29)는 Target domain의 샘플과 비슷해야 한다. ![](https://latex.codecogs.com/gif.latex?D_3)이 이렇게 만드는 역할을 하고 있다.

다음은 ![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D)다. 다음과 같이 정의된다.

![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D%20%3D%20%5Csum_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20d%28f%28x%29%2C%20f%28g%28f%28x%29%29%29%29)

함수 ![](https://latex.codecogs.com/gif.latex?f%28x%29)는 ![](https://latex.codecogs.com/gif.latex?x)의 context를 뽑는 일을 한다고 했으므로, 저 metric ![](https://latex.codecogs.com/gif.latex?d)의 값이 작아질수록 생성한 이미지와 원래 샘플의 context가 같아진다는 얘기가 된다. 즉 이미지 생성이 원본과 비슷하게 잘 된다는 이야기이다. ![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D)는 그렇게 만들어주는 역할을 한다.

다음 Loss는 ![](https://latex.codecogs.com/gif.latex?L_%7BTID%7D)이고, 다음과 같이 정의된다.

![](https://latex.codecogs.com/gif.latex?L_%7BTID%7D%20%3D%20%5Csum_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20d_2%28x%2C%20G%28x%29%29)

Metric ![](https://latex.codecogs.com/gif.latex?d_2)는 Distance function이다. 논문에서는 MSE가 사용되었다. Target domain의 샘플과 그것으로 생성한 이미지의 차이를 줄이는 역할을 한다. 

마지막 Loss는 ![](https://latex.codecogs.com/gif.latex?L_%7BTV%7D)인데. 이미지를 부드럽게(smooth)하게 만들어주는 역할을 한다. 어떤 일인지는 [위키백과 Total Variation Denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)을 참고하자. 생성한 이미지 ![](https://latex.codecogs.com/gif.latex?G%28x%29)를 이미지 내 모든 픽셀들의 집합으로 보자. ![](https://latex.codecogs.com/gif.latex?z%20%3D%20%5Bz_%7Bij%7D%5D%20%3D%20G%28x%29)로 놓고, 다음과 같이 Loss를 정의할 수 있다.

![](https://latex.codecogs.com/gif.latex?L_%7BTV%7D%28z%29%20%3D%20%5Csum_%7Bi%2C%20j%7D%20%5CBig%28%5Cbig%28z_%7Bi%2C%20j&plus;1%7D%20-%20z_%7Bij%7D%29%5E2%20&plus;%20%5Cbig%28z_%7Bi&plus;1%2C%20j%7D%20-%20z_%7Bij%7D%29%5E2%20%5CBig%29%5E%5Cfrac%7BB%7D%7B2%7D)

이 Loss가 무슨 뜻이냐면, 인접한 픽셀과 너무 큰 차이가 나는 것을 줄여주는 것이다. 바로 옆 픽셀과 바로 윗 픽셀과의 차이를 줄이려고 하는 식으로 이해할 수 있다. 논문에서는 ![](https://latex.codecogs.com/gif.latex?B%20%3D%201)로 놓고, 제곱근을 씌우는 것과 같이 사용하고 있다.

### Model Structure

위의 아이디어와 Loss 함수를 적용한 모델의 대략적인 생김새는 아래와 같다.

![](../images/UCDIG/pic1.PNG)

### Experiments

논문에서는 SVHN과 MNIST간의 Domain transfer로 실험을 했다. MNIST를 SVHN의 이미지 사이즈인 32 x 32 x 3으로 Resize했다. Grayscale의 이미지를 3번 복사해서 Depth를 3으로 만들었다.

![](../images/UCDIG/pic2.PNG)

다음은 유명한 실험인 Face to Imoji이다. CelebA 데이터셋을 가지고 실험했는데, 시중의 프로그램을 가지고 생성한 Emoji보다 더 인물의 특징을 잘 살리는 것 같은 느낌을 준다. 문근영과 같은 익숙한 얼굴들도 있다.

![](../images/UCDIG/pic3.PNG)