# pyqt-vgg16-image-classification-example
<div align="center">
  <img src="https://user-images.githubusercontent.com/55078043/229002952-9afe57de-b0b6-400f-9628-b8e0044d3f7b.png" width="150px" height="150px"><br/><br/>
  
  [![](https://dcbadge.vercel.app/api/server/cHekprskVE)](https://discord.gg/cHekprskVE)
</div>

Example of using Image classification with VGG16 in PyQt5

This is pretty much same code with <a href="https://github.com/yjg30737/pyqt-pytorch-image-classification-gui.git">pyqt-pytorch-image-classification-gui</a>.

The important difference is that this uses an already existing model. VGG16 is one of the representative models for image classification and already possesses the ability to distinguish many objects.

This shows the implementation of the VGG16 image classification function in PYQT. It displays not only the label with the highest prediction value (correct answer) but also the top 5 labels.

## Requirements
* PyQt5 >= 5.14
* torch
* torchvision
* numpy

## How to Run
1. git clone ~
2. pip install -r requirements.txt
3. python main.py

## Preview
1. Umbrella

![image](https://github.com/yjg30737/pyqt-vgg16-image-classification-example/assets/55078043/9dd6eb47-767d-4633-acb8-c307d9736733)

2. Lion

![image](https://github.com/yjg30737/pyqt-vgg16-image-classification-example/assets/55078043/4a57ea57-6e6b-4f28-9f3f-f48a8fa903f4)

## See Also

Kaggle Notebook:

https://www.kaggle.com/code/yoonjunggyu/pytorch-using-vgg16-to-image-classification
