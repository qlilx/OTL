# OTL
Clustering-based representation learning through output translation and its application to remote sensing images


ABSTRACT

In supervised deep learning, learning good representations for remote sensing images (RSI) relies on manual annotations. However in the area of remote sensing, it is hard to obtain huge amounts of labeled data. Recently, self-supervised learning shows its outstanding capability to learn representations of images, especially the methods of instance discrimination. Compared methods of instance discrimination, clustering-based methods not only view the transformations of the same image as the "positive" samples but also the similar images. In this paper we propose a new clustering-based method for representation learning. We first introduce a quantity to measure representations’ discriminativeness and from which we show that even distribution requires the most discriminative representations. This provides a theoretical insight into why evenly distributing the images works well. We notice that only the even distributions that preserve representations’ neighborhood relations are desirable. Therefore, we develop an algorithm that translates the outputs of a neural network to achieve the goal of evenly distributing the samples while preserving outputs’ neighborhood relations. Extensive experiments have demonstrated that our method can learn representations that are as good as or better than the state of the art, and that our method performs computationally efficient and robustly on various RSI datasets.

REQUIREMENTS

PyTorch 1.8.0

torchvision 0.9.0

CUDA 11.0
