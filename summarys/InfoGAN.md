# InfoGAN

### Introduction

기존 GAN은 입력 노이즈 ![](https://latex.codecogs.com/png.latex?z)를 이용하여 생성 데이터의 분포에서 이미지를 뽑아내는 모델이었습니다. 이 Generator를 학습하는 과정에서, 입력 노이즈 ![](https://latex.codecogs.com/png.latex?z)가 생성 이미지의 특징 정보를 담고 있긴 합니다. 하지만 벡터의 특정 부분이 어떤 특징을 Represent하고 있는지는 확인하기 어렵습니다. 또 논문의 저자들은, 이미지의 __회전__ 이나 __광량__ 의 조절 등 특정한 __특징__ 을 파악하여 그에 맞는 Generate를 할 수 있는 모델을 만들고자 했습니다. 이런 아이디어에서 나온 모델이 __InfoGAN__ 입니다.

InfoGAN의 목적은 간단합니다. 원래 GAN과 같은 비지도학습 방법으로 해석 가능한 '이미지의 회전 정도' 같은 특징량들을 얻어내고 싶은 것입니다. 또한 그 특징량을 적절하게 Generator에 전달하면, Generator가 그 값에 맞게 이미지를 생성하게 만들고 싶은 것입니다. 예를 들어서, 입력 노이즈 z가 '의자' 이미지를 생성하는 벡터일 때, __'오른쪽으로 30도 정도 회전'__ 을 의미하는 특징량을 같이 전달해주면, __'오른쪽으로 30도 정도 회전된 의자 이미지'__ 가 생성되는 식의 동작을 원하는 것입니다.

이를 위해서, 입력 노이즈를 두 부분으로 나누었습니다. 원래 GAN에서는 입력 노이즈 ![](https://latex.codecogs.com/png.latex?z)만을 사용했지만, InfoGAN에서는 ![](https://latex.codecogs.com/png.latex?z)와 __Latent code__ ![](https://latex.codecogs.com/png.latex?c)를 같이 사용합니다. Generator는 ![](https://latex.codecogs.com/png.latex?z)와 ![](https://latex.codecogs.com/png.latex?c)를 둘 다 사용하여 이미지를 생성하므로, ![](https://latex.codecogs.com/png.latex?G%28z%2C%20c%29)로 표현됩니다.

원래의 GAN에서도 사실 ![](https://latex.codecogs.com/png.latex?c)는 암묵적으로 도입되어 있었습니다. 단지 Generator의 분포 ![](https://latex.codecogs.com/png.latex?P_G)에 대해서 ![](https://latex.codecogs.com/png.latex?P_G%28x%7Cc%29%20%3D%20P_G%28x%29)였을 뿐입니다. 직관적으로 해석하면, 그냥 ![](https://latex.codecogs.com/png.latex?c)가 Generator의 이미지 생성과 관련이 없었다는 것입니다. InfoGAN에서는 이 문제를 해결하고 ![](https://latex.codecogs.com/png.latex?c)가 Generator의 이미지 생성에 매우 중요한 역할을 하도록 강제하였습니다.

따라서 InfoGAN의 cost는 이렇게 생각할 수 있습니다. Generator의 분포 ![](https://latex.codecogs.com/png.latex?P_G%28x%29)에서 하나의 ![](https://latex.codecogs.com/png.latex?x)를 샘플링했을 때, ![](https://latex.codecogs.com/png.latex?P_G%28c%7Cx%29)가 작은 엔트로피를 가지게 만드는 cost가 필요합니다. 이는 Generator가 이미지를 생성하는 과정에서 ![](https://latex.codecogs.com/png.latex?c)를 잊으면 안 된다는 것을 의미합니다. ![](https://latex.codecogs.com/png.latex?P_G%28c%7Cx%29)가 작은 엔트로피를 가진다는 것은 ![](https://latex.codecogs.com/png.latex?x)가 변했을 때의 ![](https://latex.codecogs.com/png.latex?c)에 대한 정보량이 최소라는 것이고, 이는 둘이 밀접한 관련을 맺고 있다는 것을 뜻합니다.

### Mutual Information Maximization

InfoGAN의 저자들은 ![](https://latex.codecogs.com/png.latex?c)와 ![](https://latex.codecogs.com/png.latex?G%28z%2C%20c%29)의 상호 의존성을 높이기 위해서, __상호 정보량 (Mutual Information)__ 의 개념을 도입합니다. 확률변수 ![](https://latex.codecogs.com/png.latex?X)와 ![](https://latex.codecogs.com/png.latex?Y)에 대해서 상호 정보량 ![](https://latex.codecogs.com/png.latex?I%28X%3BY%29)는 다음과 같이 정의됩니다.

![](https://latex.codecogs.com/png.latex?I%28X%3BY%29%20%3D%20H%28X%29%20-%20H%28X%7CY%29%20%3D%20H%28Y%29%20-%20H%28Y%7CX%29)

한마디로 __두 변수가 서로를 모를 때의 엔트로피__ 라고 보시면 될 것 같습니다.

InfoGAN의 목적은 ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)를 최대화시키는 것입니다. 상호 정보량의 정의에 따라, ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29%20%3D%20H%28c%29%20-%20H%28c%7CG%28z%2C%20c%29%29)입니다. ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)를 최대화시키려면 ![](https://latex.codecogs.com/png.latex?H%28c%7CG%28z%2C%20c%29%29)가 최소화되어야 함을 알 수 있습니다. ![](https://latex.codecogs.com/png.latex?H%28c%7CG%28z%2C%20c%29%29)는 ![](https://latex.codecogs.com/png.latex?G%28z%2C%20c%29) 의 값 하나를 알 때 ![](https://latex.codecogs.com/png.latex?c)의 엔트로피를 의미합니다. ![](https://latex.codecogs.com/png.latex?H%28c%7CG%28z%2C%20c%29%29)가 0에 가깝다는 것은 ![](https://latex.codecogs.com/png.latex?G%28z%2C%20c%29)를 알면 ![](https://latex.codecogs.com/png.latex?c)를 알 수 있다는 것이고, 이는 Generator의 생성에 ![](https://latex.codecogs.com/png.latex?c)가 중요하게 작용함을 의미합니다.

![](https://latex.codecogs.com/gif.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)를 구하던지 해야 최대화를 할 수 있을텐데, 이게 힘들다고 합니다. ![](https://latex.codecogs.com/gif.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)를 구하려면 사후 확률 ![](https://latex.codecogs.com/gif.latex?P%28c%7Cx%29)를 알아야 하는데, ![](https://latex.codecogs.com/gif.latex?x)는 ![](https://latex.codecogs.com/gif.latex?G%28z%2C%20c%29)에서 샘플링되기 때문에 ![](https://latex.codecogs.com/gif.latex?P%28c%7Cx%29) 분포를 구할 수가 없는 것입니다. 분포 내의 모든 ![](https://latex.codecogs.com/gif.latex?x)를 샘플링해야하기 떄문입니다.

따라서 논문에서는 ![](https://latex.codecogs.com/png.latex?P%28c%7Cx%29)에 근사하는 보조 분포 ![](https://latex.codecogs.com/png.latex?Q%28c%7Cx%29)를 이용해 상호 정보량 ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)가 커지도록 강제하는 __lower bound__ 를 만들어 상호 정보량의 최대화를 이루어냈습니다.

이제 보조 분포 ![](https://latex.codecogs.com/png.latex?Q)만을 이용하는 식으로 상호 정보량 ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)의 Lower bound를 정의합니다.

일단 Conditional Entropy의 정의에 따라, ![](https://latex.codecogs.com/png.latex?H%28c%7CG%28z%2C%20c%29%29)를 Expectation 연산을 이용해 전개해줍니다. Conditional Entropy의 정의는 다음과 같습니다.

![](https://latex.codecogs.com/png.latex?H%28Y%7CX%29%20%3D%20-%5Csum_x%20%5Csum_y%20p%28x%2C%20y%29%20%5Clog%20p%28y%7Cx%29)

따라서 아래와 같이 전개할 수 있습니다.

![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29%20%3D%20H%28c%29%20-%20H%28c%7CG%28z%2C%20c%29%29)

![](https://latex.codecogs.com/png.latex?%3D%20E_%7Bx%20%5Csim%20G%28z%2C%20c%29%7D%5BE_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20P%28c%27%7Cx%29%5D%5D%20&plus;%20H%28c%29)

이제 여기서 보조 분포 ![](https://latex.codecogs.com/png.latex?Q)를 도입합니다. ![](https://latex.codecogs.com/png.latex?P) 대신 ![](https://latex.codecogs.com/png.latex?Q)로 샘플링할 때의 Entropy 차이를 KL Divergence로 계산합니다. 

![](https://latex.codecogs.com/png.latex?%3D%20E_%7Bx%20%5Csim%20G%28z%2C%20c%29%7D%5BD_%7BKL%7D%28P%28%5Ccdot%7Cx%29%7C%7CQ%28%5Ccdot%7Cx%29%29%20&plus;%20E_%7Bc%27%20%5Csim%20p%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D%5D%20&plus;%20H%28c%29)

여기서 ![](https://latex.codecogs.com/png.latex?D_%7BKL%7D%28P%28%5Ccdot%7Cx%29%7C%7CQ%28%5Ccdot%7Cx%29%29)는 ![](https://latex.codecogs.com/png.latex?P)와 ![](https://latex.codecogs.com/png.latex?Q)의 Entropy 차이를 의미합니다. 또한 ![](https://latex.codecogs.com/png.latex?E_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D)는 ![](https://latex.codecogs.com/png.latex?Q)의 엔트로피를 의미합니다. ![](https://latex.codecogs.com/png.latex?P)와 ![](https://latex.codecogs.com/png.latex?Q)의 엔트로피에 대해서 ![](https://latex.codecogs.com/png.latex?P%20%3D%20D_%7BKL%7D%28P%7C%7CQ%29%20&plus;%20Q)이므로 식을 위와 같이 전개할 수 있습니다. 

우리의 목표는 ![](https://latex.codecogs.com/png.latex?P)와 ![](https://latex.codecogs.com/png.latex?Q)의 분포가 동일해지면서, 상호 정보량을 최대화하는 것입니다. 따라서 ![](https://latex.codecogs.com/png.latex?Q)를 최대화하는 방향으로 업데이트하면 둘 다 만족할 수 있습니다. 따라서 아래의 Lower bound를 만들 수 있습니다.

![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29%20%5C%20%5Cgeq%20%5C%20E_%7Bx%20%5Csim%20G%28z%2C%20c%29%7D%5BE_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D%5D%20&plus;%20H%28c%29)

![](https://latex.codecogs.com/png.latex?P)와 ![](https://latex.codecogs.com/png.latex?Q)의 KL Divergence는 Lower bound에서 사라집니다. 그 이유는, 해당 KL Divergence가 상호정보량과 관련이 없기 때문입니다. ![](https://latex.codecogs.com/png.latex?c)와 직접적인 연관이 없고 식 안에서 상수로 취급되기 때문에, 값을 Maximize해가는 Lower bound가 건들 이유가 없습니다.

![](https://latex.codecogs.com/png.latex?E_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D)가 커질수록 ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)에 대한 Lower bound가 커지고, ![](https://latex.codecogs.com/png.latex?I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)는 Lower bound 이상으로 유지되도록 강제되므로 상호정보량이 커지도록 유도할 수 있습니다. 따라서 ![](https://latex.codecogs.com/png.latex?c)가 Generator의 이미지 생성에 미치는 영향이 커집니다.

이렇게 구한 Lower bound를 ![](https://latex.codecogs.com/png.latex?L_I%28G%2C%20Q%29)로 놓겠습니다.

![](https://latex.codecogs.com/png.latex?L_I%28G%2C%20Q%29%20%3D%20E_%7Bx%20%5Csim%20G%28z%2C%20c%29%7D%5BE_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D%5D%20&plus;%20H%28c%29) 입니다.

Lower bound를 이렇게 구한 이유는 ![](https://latex.codecogs.com/png.latex?P) 대신 ![](https://latex.codecogs.com/png.latex?Q) 분포를 사용하기 위해서였는데, 식 내부에는 아직 ![](https://latex.codecogs.com/png.latex?P%28c%7Cx%29) 분포가 존재합니다. 이를 해결하기 위한 방법이 논문의 Lemma 5.1에 제시되어 있습니다. 확률변수 ![](https://latex.codecogs.com/png.latex?X), ![](https://latex.codecogs.com/png.latex?Y)에 대해서 다음 식이 성립한다고 합니다.

![](https://latex.codecogs.com/png.latex?E_%7Bx%20%5Csim%20X%2C%20y%20%5Csim%20Y%7Cx%7D%5Bf%28x%2C%20y%29%5D%20%3D%20E_%7Bx%20%5Csim%20X%2C%20y%20%5Csim%20Y%7Cx%2C%20x%27%20%5Csim%20X%7Cy%7D%5Bf%28x%27%2C%20y%29%5D)

이 법칙에 따라서 Lower bound ![](https://latex.codecogs.com/png.latex?L_I%28G%2C%20Q%29)의 식을 ![](https://latex.codecogs.com/png.latex?P%28c%7Cx%29)에 의존하지 않게 아래와 같이 바꿀 수 있습니다.

![](https://latex.codecogs.com/png.latex?L_I%28G%2C%20Q%29%20%3D%20E_%7Bx%20%5Csim%20G%28z%2C%20c%29%7D%5BE_%7Bc%27%20%5Csim%20P%28c%7Cx%29%7D%5B%5Clog%20Q%28c%27%7Cx%29%5D%5D%20&plus;%20H%28c%29) ![](https://latex.codecogs.com/png.latex?%3D%20E_%7Bc%20%5Csim%20P%28c%29%2C%20x%20%5Csim%20G%28z%2C%20c%29%7D%5B%5Clog%20Q%28c%7Cx%29%5D%20&plus;%20H%28c%29)

Lemma 5.1의 식과 ![](https://latex.codecogs.com/png.latex?L_I%28G%2C%20Q%29)의 식을 ![](https://latex.codecogs.com/png.latex?x%20%3A%20c%2C%20%5C%20%5C%20X%20%3A%20P%28c%29%2C%20%5C%20%5C%20y%20%3A%20x%2C%20%5C%20%5C%20Y%20%3A%20G%28z%2C%20c%29)
 처럼 대응시키면 쉽게 이해가 가실 겁니다.
 
### Objective Function

InfoGAN의 목적 함수는 상호정보량의 최대화를 위해 다음과 같이 정의됩니다.

![](https://latex.codecogs.com/png.latex?%5Cunderset%7BG%7D%7Bmin%7D%20%5C%20%5Cunderset%7BD%7D%7Bmax%7D%20%5C%20V_I%28D%2C%20G%29%20%3D%20V%28D%2C%20G%29%20-%20%5Clambda%20I%28c%20%5C%20%3B%20G%28z%2C%20c%29%29)

Generator는 이 목적함수를 최소화해야 하므로, 상호정보량을 최대로 만들기 위해 노력합니다. 위에서 보았듯이 상호정보량의 최대화는 Lower bound ![](https://latex.codecogs.com/png.latex?L_I%28D%2C%20Q%29)의 최대화를 의미합니다. 따라서 InfoGAN의 목적 함수의 최종 형태는 다음과 같이 정의됩니다.

![](https://latex.codecogs.com/png.latex?%5Cunderset%7BG%2C%20Q%7D%7Bmin%7D%20%5C%20%5Cunderset%7BD%7D%7Bmax%7D%20%5C%20V_%7B%5Ctextnormal%7BInfoGAN%7D%7D%28D%2C%20G%2C%20Q%29%20%3D%20V%28D%2C%20G%29%20-%20%5Clambda%20L_I%28G%2C%20Q%29)

### Model Training

분포 ![](https://latex.codecogs.com/png.latex?Q)를 NN으로 구성합니다. ![](https://latex.codecogs.com/png.latex?Q)는 Discriminator 모델과 모든 Convolution Layer를 공유하고 ![](https://latex.codecogs.com/png.latex?Q)만의 Fully-Connected Layer를 가지고 있습니다. 이 레이어는 조건부 분포 ![](https://latex.codecogs.com/png.latex?Q%28c%7Cx%29)를 의미합니다.

이런 구조를 가지고 있는 덕분에, 논문에서는 GAN 학습시에 덤으로 따라오는 학습을 할 수 있다고 표현하고 있습니다. 실제로 모든 Convolution을 공유하기 때문에 두 개의 모델을 구성하는 데에 필요한 레이어 개수가 한 개일때와 거의 동일합니다. 학습 방식 또한 DCGAN과 동일한 구조를 가지기 때문에, Latent code를 학습하는데 필요한 추가적인 컴퓨팅이 적습니다.

### Experiments

Latent code의 학습은 확실한 효과를 보여주었습니다.

![](../images/InfoGAN/InfoGAN1.PNG)

MNIST 이미지를 생성하는 GAN 모델에서 Latent code ![](https://latex.codecogs.com/png.latex?c_1%2C%20c_2%2C%20c_3)를 덤으로 학습시켰습니다. ![](https://latex.codecogs.com/png.latex?c_1)은 MNIST의 숫자 종류를 의미하는 Latent code입니다. 0~9 사이의 숫자를 의미하죠. 그림 (a)를 보면 ![](https://latex.codecogs.com/png.latex?c_1)을 바꿔가면서 이미지를 생성하여 0부터 9 사이의 각기 다른 숫자들을 생성하고 있는 모습을 볼 수 있습니다. 그림 (b)는 Latent code를 학습하지 않은 일반 GAN에서 Latent Code를 바꿔가며 생성한 결과를 보여줍니다. 

![](https://latex.codecogs.com/png.latex?c_2)는 숫자의 회전을 의미합니다. 그림 (c)를 보면, ![](https://latex.codecogs.com/png.latex?c_2)를 -2에서 2 사이로 바꿔가면서 이미지의 변화를 관찰한 것을 볼 수 있습니다. ![](https://latex.codecogs.com/png.latex?c_2)가 커질수록 숫자가 오른쪽으로 회전하고 있는 모습을 볼 수 있습니다.

![](https://latex.codecogs.com/png.latex?c_3)은 숫자의 너비를 의미합니다. 그림 (d)를 보면, ![](https://latex.codecogs.com/png.latex?c_3)를 -2에서 2 사이로 바꿔가면서 이미지의 변화를 관찰한 것을 볼 수 있습니다. ![](https://latex.codecogs.com/png.latex?c_3)가 커질수록 숫자가 넓어지는 모습을 볼 수 있습니다.

아래의 두 번째 실험 결과도 위와 같이 Latent code가 잘 학습된 것을 볼 수 있습니다.

![](../images/InfoGAN/InfoGAN2.PNG)
