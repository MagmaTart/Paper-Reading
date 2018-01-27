# UCDIG

### Introduction

Source domain ![](https://latex.codecogs.com/gif.latex?S)�� Target domain ![](https://latex.codecogs.com/gif.latex?T)�� ���ؼ�, � ���� ![](https://latex.codecogs.com/gif.latex?x)�� ���� Generator Function ![](https://latex.codecogs.com/gif.latex?G%20%3A%20S%20%5Crightarrow%20T)�� ���ǵ� ��, ![](https://latex.codecogs.com/gif.latex?G%28x%29)�� ![](https://latex.codecogs.com/gif.latex?S)���� ���� ������ ![](https://latex.codecogs.com/gif.latex?T) �������� �������� ��ȯ�ϴ� �۾��� �Ѵ�. �� ���� ������ �̸� �����ϴ� GAN ��Ʈ��ũ�� ����� ���̴�. ![](https://latex.codecogs.com/gif.latex?G%28x%29)�� ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20S)�̵� ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20T)�̵� ������� ![](https://latex.codecogs.com/gif.latex?T) �������� �̹����� �������� ���ϴ� ������ ���鵵�� �н��ȴ�. �׸��� �̷��� �н��Ǵ� ��Ʈ��ũ�� __Domain Transfer Network(DTN)__ �̶�� �Ѵ�.

Source domain ![](https://latex.codecogs.com/gif.latex?S)���� � ���� ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_S)�� ���� ���� �׵� ����(i.i.d)�� ���ø��� ���̺� ���� ���� ��Ʈ ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bs%7D)�� �־�����, Target domain�� ��Ʈ ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bt%7D) ���� ![](https://latex.codecogs.com/gif.latex?T)���� ���� ������� ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_T)�� ���� �־�����. �Լ� ![](https://latex.codecogs.com/gif.latex?f)�� �־����µ�, �� �Լ��� �Է����� � �������� ������ �������� ������ ![](https://latex.codecogs.com/gif.latex?T)�� �����ϴ� �Լ��̴�. �׸��� � weight�� ![](https://latex.codecogs.com/gif.latex?%5Calpha)�� metric ![](https://latex.codecogs.com/gif.latex?d)�� ���� �־��� ��,  DTN�� �ּ�ȭ�ϰ��� �ϴ� ���� �Լ��� ������ ����.

![](https://latex.codecogs.com/gif.latex?R%20%3D%20R_%7BGAN%7D%20&plus;%20%5Calpha%20R_%7BCONST%7D)

![](https://latex.codecogs.com/gif.latex?R_%7BGAN%7D%20%3D%20%5Cunderset%7BD%7D%7Bmax%7D%20%5C%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_S%7D%20%5Clog%5B1%20-%20D%28G%28x%29%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_T%7D%20%5Clog%5BD%28x%29%5D)

![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20%5Cmathcal%7BD%7D_S%7D%20d%28f%28x%29%2C%20f%28G%28x%29%29%29)

![](https://latex.codecogs.com/gif.latex?R_%7BGAN%7D)�� GAN�� ���� �Լ��� Domain Transfer ������ �°� ������Ų ���̴�. Discriminator ![](https://latex.codecogs.com/gif.latex?D)�� �Է��� Target domain�� ������ �� 1�� �з��ϵ��� �н���Ų��. ���� �� �׿����� Source domain ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_S)���� ���ø��� ![](https://latex.codecogs.com/gif.latex?x)�� Discriminator�� �־��� ��, Target domain�� ���÷� �ν��ؼ� 1�� �з��ϵ��� �����ϰ� �ִ�. �� ���� ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BD%7D_T)���� ���ø��� Target domain�� ������ �� �з��ϵ��� ����� ���̴�.

![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D)�� f-constancy�� �����ϴ� Term�̴�. �Լ� ![](https://latex.codecogs.com/gif.latex?f)�� �Է��� Source domain���� ���� ������ ���� Generator�� ������ ��, �� ����� ���̸� �ּ�ȭ�Ϸ��� �ϰ� �ִ�. �������� metric ![](https://latex.codecogs.com/gif.latex?d)�� MSE�� ����ϰ� �ִ�. �� ���� ��, ![](https://latex.codecogs.com/gif.latex?R_%7BCONST%7D)�� �۾������� �Լ� ![](https://latex.codecogs.com/gif.latex?f)�� ���� ![](https://latex.codecogs.com/gif.latex?x)�� ![](https://latex.codecogs.com/gif.latex?G%28x%29)�� �������ٴ� ���� �ȴ�. �Լ� ![](https://latex.codecogs.com/gif.latex?f)�� ������ ![](https://latex.codecogs.com/gif.latex?T)�� ���ؼ� ���ǵǾ��ֱ� ������, �̴� Source domain�� ���ð� Generator�� ��ģ ������ '�ǹ�', �� Context�� ���ٰ� �ν��ϰ� �ִٴ� ������ �ؼ��� �� �ִ�.

### Loss function

DTN�� �ٽ��� �ٸ� ��Ʈ��ũ�� �׷����� Loss �Լ���. ���� DTN������ Generator ![](https://latex.codecogs.com/gif.latex?G)�� ![](https://latex.codecogs.com/gif.latex?g%20%5Ccirc%20f)�� ���� �ִ�. ![](https://latex.codecogs.com/gif.latex?f)�� �Էµ� �̹������� Context�� encode�ϴ� �Լ��̰�, ![](https://latex.codecogs.com/gif.latex?g)�� �� context�� �������� �ٽ� ������ Generate�ϴ� �Լ��̴�.

DTN�� Loss �Լ��� Discriminator�� ���� ![](https://latex.codecogs.com/gif.latex?L_D)�� Generator�� ���� ![](https://latex.codecogs.com/gif.latex?L_G) �ѷ� �̷���� �ִ�. ���� ![](https://latex.codecogs.com/gif.latex?L_D)�� ������ ���� ���ǵȴ�.

![](https://latex.codecogs.com/gif.latex?L_D%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20%5Clog%20D_1%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_2%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_3%28x%29)

![](https://latex.codecogs.com/gif.latex?D)�� ���� �з��� �����ϴ� �Լ��̰�, ![](https://latex.codecogs.com/gif.latex?D_i%28x%29)�� �Է� ![](https://latex.codecogs.com/gif.latex?x)�� Ŭ���� ![](https://latex.codecogs.com/gif.latex?i)�� ���� Ȯ���� ��ȯ�Ѵ�. ���� Loss ![](https://latex.codecogs.com/gif.latex?L_D)�� 

- ���� ![](https://latex.codecogs.com/gif.latex?x)�� ������ �̹������� ![](https://latex.codecogs.com/gif.latex?x)�� Source domain���� ���Դ��� (![](https://latex.codecogs.com/gif.latex?D_1))
- ���� ![](https://latex.codecogs.com/gif.latex?x)�� �ļ��� �̹������� ![](https://latex.codecogs.com/gif.latex?x)�� Target domain���� ���Դ��� (![](https://latex.codecogs.com/gif.latex?D_2))
- ���� ![](https://latex.codecogs.com/gif.latex?x)�� Target domain���� ���Դ��� (![](https://latex.codecogs.com/gif.latex?D_3))

�� ������ �̷���� �ִٰ� �� �� �ִ�.

������ Generator�� Loss ![](https://latex.codecogs.com/gif.latex?L_G)�̴�. ![](https://latex.codecogs.com/gif.latex?L_G%20%3D%20L_%7BGANG%7D%20&plus;%20%5Calpha%20L_%7BCONST%7D%20&plus;%20%5Cbeta%20L_%7BTID%7D%20&plus;%20%5Cgamma%20L_%7BTV%7D)�� ���ǵȴ�. ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%2C%20%5Cbeta%20%2C%20%5Cgamma)�� Weight�̴�.
���� ![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D)���� ���캸��.

![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D%20%3D%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20%5Clog%20D_3%28g%28f%28x%29%29%29%20-%20%5Cmathbb%7BE%7D_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20%5Clog%20D_3%28g%28f%28x%29%29%29)

���� �ǹ̴� �����ϴ�. ![](https://latex.codecogs.com/gif.latex?L_%7BGANG%7D)�� �ּ�ȭ�Ϸ���, ���� ![](https://latex.codecogs.com/gif.latex?x)�� Source domain���� ���ø���� Target domain���� ���ø����������, ![](https://latex.codecogs.com/gif.latex?x)�κ��� ������ �̹��� ![](https://latex.codecogs.com/gif.latex?g%28f%28x%29%29)�� Target domain�� ���ð� ����ؾ� �Ѵ�. ![](https://latex.codecogs.com/gif.latex?D_3)�� �̷��� ����� ������ �ϰ� �ִ�.

������ ![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D)��. ������ ���� ���ǵȴ�.

![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D%20%3D%20%5Csum_%7Bx%20%5Cin%20%5Cmathbf%7Bs%7D%7D%20d%28f%28x%29%2C%20f%28g%28f%28x%29%29%29%29)

�Լ� ![](https://latex.codecogs.com/gif.latex?f%28x%29)�� ![](https://latex.codecogs.com/gif.latex?x)�� context�� �̴� ���� �Ѵٰ� �����Ƿ�, �� metric ![](https://latex.codecogs.com/gif.latex?d)�� ���� �۾������� ������ �̹����� ���� ������ context�� �������ٴ� ��Ⱑ �ȴ�. �� �̹��� ������ ������ ����ϰ� �� �ȴٴ� �̾߱��̴�. ![](https://latex.codecogs.com/gif.latex?L_%7BCONST%7D)�� �׷��� ������ִ� ������ �Ѵ�.

���� Loss�� ![](https://latex.codecogs.com/gif.latex?L_%7BTID%7D)�̰�, ������ ���� ���ǵȴ�.

![](https://latex.codecogs.com/gif.latex?L_%7BTID%7D%20%3D%20%5Csum_%7Bx%20%5Cin%20%5Cmathbf%7Bt%7D%7D%20d_2%28x%2C%20G%28x%29%29)

Metric ![](https://latex.codecogs.com/gif.latex?d_2)�� Distance function�̴�. �������� MSE�� ���Ǿ���. Target domain�� ���ð� �װ����� ������ �̹����� ���̸� ���̴� ������ �Ѵ�. 

������ Loss�� ![](https://latex.codecogs.com/gif.latex?L_%7BTV%7D)�ε�. �̹����� �ε巴��(smooth)�ϰ� ������ִ� ������ �Ѵ�. � �������� [��Ű��� Total Variation Denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)�� ��������. ������ �̹��� ![](https://latex.codecogs.com/gif.latex?G%28x%29)�� �̹��� �� ��� �ȼ����� �������� ����. ![](https://latex.codecogs.com/gif.latex?z%20%3D%20%5Bz_%7Bij%7D%5D%20%3D%20G%28x%29)�� ����, ������ ���� Loss�� ������ �� �ִ�.

![](https://latex.codecogs.com/gif.latex?L_%7BTV%7D%28z%29%20%3D%20%5Csum_%7Bi%2C%20j%7D%20%5CBig%28%5Cbig%28z_%7Bi%2C%20j&plus;1%7D%20-%20z_%7Bij%7D%29%5E2%20&plus;%20%5Cbig%28z_%7Bi&plus;1%2C%20j%7D%20-%20z_%7Bij%7D%29%5E2%20%5CBig%29%5E%5Cfrac%7BB%7D%7B2%7D)

�� Loss�� ���� ���̳ĸ�, ������ �ȼ��� �ʹ� ū ���̰� ���� ���� �ٿ��ִ� ���̴�. �ٷ� �� �ȼ��� �ٷ� �� �ȼ����� ���̸� ���̷��� �ϴ� ������ ������ �� �ִ�. �������� ![](https://latex.codecogs.com/gif.latex?B%20%3D%201)�� ����, �������� ����� �Ͱ� ���� ����ϰ� �ִ�.

### Model Structure

���� ���̵��� Loss �Լ��� ������ ���� �뷫���� ������� �Ʒ��� ����.

![](../images/UCDIG/pic1.PNG)

### Experiments

�������� SVHN�� MNIST���� Domain transfer�� ������ �ߴ�. MNIST�� SVHN�� �̹��� �������� 32 x 32 x 3���� Resize�ߴ�. Grayscale�� �̹����� 3�� �����ؼ� Depth�� 3���� �������.

![](../images/UCDIG/pic2.PNG)

������ ������ ������ Face to Imoji�̴�. CelebA �����ͼ��� ������ �����ߴµ�, ������ ���α׷��� ������ ������ Emoji���� �� �ι��� Ư¡�� �� �츮�� �� ���� ������ �ش�. ���ٿ��� ���� �ͼ��� �󱼵鵵 �ִ�.

![](../images/UCDIG/pic3.PNG)