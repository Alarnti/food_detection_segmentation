# Food detection with segmentation

ToDo list:
- [x] add initial model from Pytorch
- [ ] train model on Colab or my machine
  - [ ] check quality for different settings
    - [ ] raw (no warmup, augs, adam, backbone frozen)
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
