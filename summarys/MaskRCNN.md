# Mask R-CNN

#### Introduction
Mask R-CNN�� ���� Two-Stage Detector �𵨰��� State-of-the-art ���Դϴ�. �� �� ���п� Instance Segmentation�̶�� Detection�� ���ο� �о߰� ��ô�� ��ŭ, ���� ������ �޾Ҵ� ���Դϴ�.

�� ���� ������ Object Instance Segmentation�� ���� �����ϰ� ������ �����ӿ�ũ�� ����� ���̾��ٰ� �մϴ�. Instance Segmentaion�� �Է� �̹��� ������ �� Object�� ã�Ƴ��� ���ÿ� ���� ����Ƽ�� Segmentation Mask�� �����ִ� �۾��� �ǹ��մϴ�. �̰��� ������ ���ؼ�, Faster R-CNN�� ���� �ִ� Bounding Box Regressor�� ���������� ���ư��� Object Mask Predictor�� ���̰ڴٴ� ���̵� ���÷Ƚ��ϴ�. �׸��� ���� ������ ������, �� ���� Faster R-CNN�� ������ �߰������� ���� ���� ���̴� ������� ���� �� �ְ�, �߰����� Overhead�� ���� ���� Faster R-CNN�� 5fps�� �ӵ��� ������ �����Ѵٰ� �մϴ�. �� Mask R-CNN�� �̿��ϸ�, Segmentation�� ����Ǵ� Ư¡�� �����ؼ� ����� ��ü ��� ��Ƴ��� ���� �پ��� Ȱ���� �����ϴٰ� �մϴ�. �� �� ����, COCO�� ��� Task (Instance Segmentation, Bounding-Box Object Detection, Person Keypoint Detection)���� �ְ��� ����� �����־����ϴ�.

#### Mask R-CNN
Mask R-CNN�� ��Ʈ��ũ ������ �����մϴ�. ���� Faster R-CNN �𵨿���, �� RoI�� ���� Segmentation Mask�� �����ϴ� ��Ʈ��ũ ����(branch)�� Bounding Box�� �����ϴ� ������ ���������� ���ÿ� ���ư����� �߰��� ���Դϴ�. ���⼭ Mask branch�� ���� FCN(Fully Convolutional Network) ��Ʈ��ũ��, �ȼ� ������ Segmentation�� �����մϴ�.

Faster R-CNN�� ������ Ȯ���� �����ϵ��� �����ϰ� �����Ǿ� �ֱ� ������, ���ٸ� Computational Cost�� ���� ���� ���� ������ �� �־��ٰ� �մϴ�. ������ ���� �𵨿��� �߰������� Masking�� �����ϴ� ���� ������ �־����ϴ�. Faster R-CNN�� ��Ʈ��ũ�� Input�� Output���� �ȼ� ������ ������ �ʿ� �������ϴ�. Bounding Box�� ã�� Classification�� �ϸ� �׸��̾��� �����Դϴ�. RoI Pooling�� ����� ���� �׷��ѵ�, �̴� Ư¡�� �����ϴ� �������� ����ȭ(quantization)�� ����ŵ�ϴ�. �׷��� ����ȭ�� ������ �� �ִ� ����� RoiAlign�� ����ؼ� �����Ͽ��ٰ� �մϴ�.

���⼭ RoIAlign�� ���� �˾ƺ��� ���ڽ��ϴ�.

���� Faster R-CNN���� RoI pooling�� �� RoI���� 7 x 7�� ���� ���� ũ���� Feature Map�� �����ϱ� ���� ���˴ϴ�. Backbone Network�� ��ġ�鼭 �Է� �̹����� ���� Feature Map�� ����� �پ����� ������, RoI�� ũ��� ���� �Ҽ����� ���� Floating Number�� �Ǿ� �ֽ��ϴ�. �װ� ����ȭ�ؼ� RoI�� �ٽ� ����� �������� quantization�� �Ͼ�� �˴ϴ�. �Ǽ��� ������ �ٲٴ� �������� �ȼ� ������ ��Ȯ�� ������ ���õǴ°̴ϴ�. �� RoI�� �� ���� ���� bin���� ���������� Max Pooling ���� ������� ������ �����µ�, �� �������� quantization�� �� �ѹ� �����˴ϴ�. ���� ��� 20 x 20 ũ���� RoI���� 7 x 7 ũ���� Feature Map�� �����Ϸ��� �ϸ�, 20 / 7 = �� 2.86���� ���� �� �������� �ʱ� ������ ��¿ �� ���� ��� �ȼ����� �ֺ� �ȼ���� ���յǴ� ���� �߻��ϰ� �˴ϴ�.

���� ���, �Է� �̹������� Ư�� �ȼ��� ��ġ�� ![](https://latex.codecogs.com/gif.latex?x)���� Feature map�� Stride�� 16, �� ũ�Ⱑ �Է� �̹����� 1/16���� �پ����ٸ�, RoI Pooling������ ![](https://latex.codecogs.com/gif.latex?%5Bx/16%5D)���� ����߾����ϴ�. ![](https://latex.codecogs.com/gif.latex?%5Bn%5D)�� ![](https://latex.codecogs.com/gif.latex?n)�� �ݿø��� �ǹ��մϴ�.
�̷� quantization�� RoI�� ����� Feature ���̰� ���� �ʰ�(misalignment) ����ϴ�. ���׸��� ������ ��ȭ�� ������(robust) �з� �������� ũ�� ����� ��������, �ȼ� ������ Mask�� �����ϴ� ���������� ū ������ �ҷ��Խ��ϴ�.

�Ʒ��� ������ ���ô�. �Ʒ��� �� ������ Youtube - Ardian Umam ���� ���ǿ��� �����Խ��ϴ�. [���� ���� Youtube ��ũ](https://www.youtube.com/watch?v=XGi-Mz3do2s)

![](../images/MaskRCNN/RoIPooling.png)

800 x 800 ũ���� �Է� �̹����� Backbone Network�� VGG�� ��ġ�鼭 Stride 32�� Feature Map ����� �������ϴ�. �� �������� ó���� 665 x 665 ũ�⸦ ������ �ִ� RoI�� (665/32) x (665/32) ũ�⸦ ������ �Ǿ��µ�, RoI�� ũ��� �Ǽ��� �� �����Ƿ� �ݿø��� �ؼ� 20���� ũ�⸦ ���ߴ� ����� ���Դϴ�. �� �̷��� ������� 20 x 20 ũ���� RoI�� 7 x 7�� Feature Map���� ����� �������� �� bin���� ����ִ� �ȼ��� ������ �޶����� �ǰ�, ��������� ��� �ȼ��� ������ �����ϰ� ������ ���� �� ���� �˴ϴ�. �̸� quantization ��ٰ� �θ��� ���Դϴ�.

RoIAlign�� �ݿø� ������ ������ �������� �ȼ��� ������ �ʰ�, bin ���� �ȼ� ������ �Ǽ����� ���Ӵϴ�. �׳� RoI Pooling�� bin ���� �ȼ��� ���������� ������ �׳� Max Pooling�� ���������� RoIAlign�� �׷��� �����Ƿ�, �� bin ������ �����ϰ� 4���� ������ �̾Ƴ��� �� ������ ��Ȯ�� Input Feature������ ���� Bilinear Interpolation���� ���ϴ� �۾��� ��ġ�� �˴ϴ�. �̷��� �ϸ� bin �� �ȼ��� ������ �������� �ƴϴ���, ������ó�� Pooling�� �� �� �ְ� �˴ϴ�.

![](../images/MaskRCNN/RoIAlign.png)

���� ������ �Ʊ�� ���� �����̵忡�� ������ �����Դϴ�. RoI Pooling���� ������ ���ڵ��� ��� ������ �����ִ� �Ͱ� �޸�, ���⼭�� ���� �Ǽ��� ��� �����ϰ� �ֽ��ϴ�. ������ �Ǵ� Pooling �κ��� ��� bin�� ���� ������ value�� ������ ������ �� �ֵ��� Bilinear Interpolation���� Feature�� value�� ���ϸ鼭 ��� �ȼ��� ��ġ�� ������ �ִ��� �����ǵ��� �ϰ� �ֽ��ϴ�.

���� �ٽ� ���� ������ ���ƿ��ڽ��ϴ�. 

FCN�� �ȼ� ������ Segmentation�� Classification�� ���ÿ� �����ϴ� �Ͱ� �޸�, Mask R-CNN�� ����ũ ������ Ŭ���� ������ �и��߽��ϴ�. �� Ŭ������ ���ؼ� ���������� Binary Mask�� �����ϰ� �˴ϴ�. FCN�� ����� ![](https://latex.codecogs.com/gif.latex?K)���� ����ũ�� �� RoI���� �̾Ƴ��ϴ�. ![](https://latex.codecogs.com/gif.latex?K)���� �̹��� Ŭ������ ������ ��, ![](https://latex.codecogs.com/gif.latex?m%20%5Ctimes%20m) �ػ��� Binary Mask�� ![](https://latex.codecogs.com/gif.latex?K)�� �����ϴ� Mask Branch�� ��� ���� ���� ![](https://latex.codecogs.com/gif.latex?Km%5E2) �Դϴ�.

��, �� RoI�� ���ؼ� Multi-task Loss�� ����Ͽ����ϴ�. ���� �Ʒ��� �����ϴ�.

![](https://latex.codecogs.com/gif.latex?L%20%3D%20L_%7Bcls%7D%20&plus;%20L_%7Bbox%7D%20&plus;%20L_%7Bmask%7D)

Classification Loss�� ![](https://latex.codecogs.com/gif.latex?L_%7Bcls%7D)�� Bounding Box Loss�� ![](https://latex.codecogs.com/gif.latex?L_%7Bbox%7D)�� Fast R-CNN������ �����ϰ� �����˴ϴ�.

�ȼ����� Sigmoid�� �����ϱ� ���ؼ�, Mask Prediction Loss�� ![](https://latex.codecogs.com/gif.latex?L_%7Bmask%7D)�� �ȼ� ���� Binary Cross-entropy Loss�� ������� �̷�����ϴ�. �ش� RoI�� Ground-truth Ŭ������ ![](https://latex.codecogs.com/gif.latex?k)��° Ŭ�������, ![](https://latex.codecogs.com/gif.latex?L_%7Bmask%7D)�� ![](https://latex.codecogs.com/gif.latex?k)��° ����ũ������ ���ǵǰ�, ������ ����ũ ��µ��� Loss�� ������ ��ġ�� �ʽ��ϴ�.
![](https://latex.codecogs.com/gif.latex?L_%7Bmask%7D)�� �̷� �������� Ŭ������ ������� ��� ����ũ�� ������ �� �ְ� ������ݴϴ�. Classification Branch�� ��� ����ũ�� �����ϱ� ���� Ŭ������ �����ϴ� �뵵�� ���˴ϴ�. Mask R-CNN������ �̷��� Mask�� Class�� Prediction�� �и��߽��ϴ�.

#### Model Structure / Training
Backbone Network�δ� ResNet-FPN�� ����Ѵٰ� �մϴ�. ResNet ������ ����� �Ŀ� Feature�� FPN �������� �����Ѵٰ� ���� �� �� �����ϴ�. Mask R-CNN�� ��Ʈ��ũ ������ ���� �Ʒ��� ����, ���� Faster R-CNN���� Mask�� Prediction�ϴ� Branch�� FCN ������ �߰��Ǿ� �ִ� ����� ���� �� �ֽ��ϴ�.

![](../images/MaskRCNN/MaskrcnnStructure.png)

Masking�� Target�� Ground-truth Mask�� RoI���� Intersection�Դϴ�. Mask Branch�� ���Ϳ� �ڽ��� ������ Mask�� ���߱� ���� Training�մϴ�.

#### Result
�Ʒ��� ����� �� Cityscape �����ͼ� ���� �̹����鿡 ���ؼ� Prediction�� ������ ����Դϴ�.

![](../images/MaskRCNN/MaskrcnnResult.png)

FPN ������ Backbone�� ���ִ� ���п� ���� ũ���� ��ü�� �� �����ϰ�, �� Masking�� �� �ϰ� �ִ� ����� �� �� �ֽ��ϴ�.

�Ʒ��� COCO �����ͼ� ���� �̹����鿡 ���� ����Դϴ�.

![](../images/MaskRCNN/cocoresult.png)

