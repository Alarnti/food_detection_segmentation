# Food detection with segmentation

[Food competition](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022)

ToDo list:
- [x] add initial model from Pytorch
- [ ] train model on Colab or my machine
  - [ ] check quality for different settings
    - [x] raw (no warmup, augs, adam, backbone frozen)
    - [ ] with warmup
    - [ ] with augs
    - [ ] with different optimizers 
- [x] create proper validation | created mAP from torchmetrics
- [ ] add mlflow
- [ ] add tests for code
- [ ] take a look on dvc
- [ ] find a solution for demo in web browser
- [ ] put model on a server with working api (TFServing)
- [ ] write web page for inference on user's images
- [ ] optimize network
- [ ] list other potential architectures 

[colab train notebook](https://colab.research.google.com/drive/1vkrpdBNqSEyHxmXjoIJXOsRYEKS9L_Ls?usp=sharing)
