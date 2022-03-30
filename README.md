# [AnoDFDNet: A deep feature difference network for anomaly detection](https://arxiv.org/abs/2203.15195)
![image](https://user-images.githubusercontent.com/79884379/158046455-2465bd04-b390-4aaf-a0c8-56a4a239b32f.png)

Python 3.7.0  
Pytorch 1.6.0  
Visdom 0.1.8.9  
Torchvision 0.7.0

Before training and test, please revise the setting in constants.py
## Training
```
python -m visdom.server
python train.py
```
## Test
```
python test.py
```
## Experiment Results
![image](https://user-images.githubusercontent.com/79884379/160744906-b3d58e54-3497-4a1c-a791-88dffd7fb673.png)
![image](https://user-images.githubusercontent.com/79884379/160744916-fd3940e3-3820-4755-a670-e2774b1ad8ec.png)
![image](https://user-images.githubusercontent.com/79884379/160744938-9058b0c1-1873-4179-9868-82a43bdf4341.png)
![image](https://user-images.githubusercontent.com/79884379/160745091-c56901b0-e1d7-4187-a3b4-0fdf3947c4e1.png)
