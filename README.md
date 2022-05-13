# vto-learning-based-animation-pytorch
A pytorch re-implementation of paper "Learning-Based Animation of Clothing for Virtual Try-On"

Official tensorflow implementation:    https://github.com/isantesteban/vto-learning-based-animation

Something different 
* learning the fit regressor and wrinkle regressor separately 
* add more body shapes when training the fit regressor
* simply use the shape blend shapes from SMPL body(instead of using the result of fit regressor) when training the wrinkle regressor 

![image](https://user-images.githubusercontent.com/37951601/168271080-6716d739-bb11-4592-8192-15a4fd148d2e.png)
![image](https://user-images.githubusercontent.com/37951601/168271146-62120f48-37da-4904-aba7-a943260251ff.png)


TODO: 
- [ ]  add support for TBPTT 
- [ ]  release more body shape data
