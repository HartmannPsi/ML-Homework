#set text(font: ("Libertinus Serif", "Noto Serif CJK SC"))
// #set math.font("libertinus")

#align(center, [= Efficient Semi-Supervised Learning for Handwritten Digit Classification using FixMatch])

#align(center, [== Abstract])

We explore the use of FixMatch@fixmatch, a state-of-the-art semi-supervised learning framework, for handwritten digit classification under limited labeled data. Our setup involves a dataset containing only 200 labeled MNIST@mnist images (20 images for each category) and 10,000 unlabeled images with images out of distribution (80% MNIST, 20% EMNIST-Letters@emnist). We evaluate several lightweight image classification models, including CNN, ResNet18@resnet, MobileNetV2@mobilenet, ShuffleNetV2@shufflenet, and Vision Transformer (ViT)@vit, under a unified FixMatch training protocol. Empirical results show that with proper data augmentation and training strategies, ResNet18 achieves over 99% test accuracy, demonstrating FixMatch's strong potential under extreme low-resource settings. We further analyze the impact of model architecture, data augmentation, and optimizer choices on training efficiency and generalization.

== 1. Introduction

=== 1.1 Background

Supervised deep learning models typically require large labeled datasets to achieve high accuracy. However, in many practical applications such as medical imaging, OCR, or low-resource language processing, labeled data is extremely limited. Semi-supervised learning (SSL) aims to leverage unlabeled data to improve generalization by combining supervised and unsupervised losses. Among SSL methods, FixMatch@fixmatch has emerged as a simple yet powerful framework that combines consistency regularization with confidence-based pseudo-labeling.

=== 1.2 Task Overview

In this work, we explore to apply FixMatch to the task of handwritten digit recognition on MNIST@mnist. We simulate a low-resource environment with only very few labeled data for each category but a large set of unlabeled ones. To simulate the noisy working environment in real-time tasks, we added EMNIST-letters@emnist images into the unlabeled dataset. Thus the model must learn meaningful features by utilizing a large set of unlabeled images. Our goal is to explore how model architecture and training strategies affect final performance under such constraints.

== 2. Methodology

=== 2.1 Dataset Setup

Labeled Data: 200 MNIST images, 20 per class.

Unlabeled Data: 10,000 images, composed of 8000 MNIST + 2000 EMNIST-Letters.

Test Data: 10,000 MNIST test images.

We use standard preprocessing (resizing to 28x28 and normalization). For ViT, inputs are interpolated to 224x224.

=== 2.2 Models Evaluated

We benchmark the following architectures:

SimpleCNN: Baseline convolutional model with two Conv-Pool blocks.

ResNet18@resnet: Standard residual network with modified input for grayscale.

MobileNetV2@mobilenet: Lightweight model with depthwise separable convolutions.

ShuffleNetV2@shufflenet: Highly efficient model optimized for mobile.

ViT-Tiny@vit: Transformer-based image classifier adapted for grayscale.

=== 2.3 FixMatch Framework

On a mixed dataset with batch size $B$ for labeled data and $mu B$ for unlabeled data, FixMatch minimizes a combined loss which is defined as follows.@fixmatch

Supervised Loss: Cross-entropy on labeled data.

$
  cal(l)_s := 1 / B sum_(b=1)^B H(p_b,p_m (y|x_b)).
$

Unsupervised Loss: Confidence-filtered cross-entropy on strongly-augmented unlabeled data with pseudo-labels generated from weakly-augmented views. The hyperparameter $tau$ is the confidence threshold for pseudo-labeling.

$
  cal(l)_u := 1 / (mu B) sum_(b=1)^(mu B) II(max(q_b)>tau)H(hat(q)_b,q_b).
$

The total loss is a weighted sum of the supervised and unsupervised losses, which is defined as:
$
  cal(l) := cal(l)_s + lambda dot cal(l)_u
$

where $lambda$ is a hyperparameter controlling the balance between the two losses.

According to the suggestions@fixmatch in Fixmatch paper, we choose $lambda = 1.0$ and conventional SGD optimizer with momentum instead of Adam@adam. We also use a cosine learning rate decay schedule@coslr.

We enhance the standard FixMatch protocol by:

- Applying RandAugment@randaug for strong augmentation.

- Introducing dynamic confidence thresholding.

- Optionally disabling horizontal flip, which is detrimental on MNIST@mnist.

We use PyTorch@pytorch for implementation.

== 3. Experiments and Results

=== 3.1 Training Configuration

Optimizer: SGD with momentum=$0.9$, weight_decay=$5times 10^(-4)$, nesterov=True;

Learning Rate: $0.03$ with cosine decay;

Batch Sizes: $64$ for labeled, $128$ for unlabeled;

Epochs: $1000$ unless noted otherwise;

Dynamic Threshold Strategy: Caculate the threshold $tau$ based on current epoch $n$ and hyperparameter $k$, $tau_max$ and $tau_min$:

$
  tau = max(tau_max, space tau_min +(1-tau_min)dot (1-e^(-n / k)))
$

in which $tau_max=0.95$, $tau_min=0.5$, $k = 50$.

Other hyperparameter settings can be found in the code repository.

=== 3.2 Performance Comparison

The following table summarizes the performance of different models on the MNIST test set after 1000 epochs of training. The accuracy is reported as the percentage of correctly classified images.

#table(
  columns: 3,
  stroke: 0pt + black,
  [Model], [Parameters], [Accuracy (%)],
  [SimpleCNN], [\~225K], [\~97.3],
  [ResNet18], [\~11M], [\~*99.0*],
  [MobileNetV2], [\~2.2M], [\~98.2],
  [ShuffleNetV2], [\~351K], [\~97.0],
  [ViT-Tiny], [\~5.4M], [-],
)

In our experiment, ResNet18 outperforms all other models in accuracy and convergence speed. ShuffleNet underperforms in this small-scale setting, and ViT without pretraining fails to converge within 1000 steps.

=== 3.3 Impact of Augmentation and Flip Removal

Using RandAugment and 4x augmentation per labeled sample improves early convergence: ResNet18 reaches 95% accuracy in only 100 epochs.

Removing horizontal flip improves accuracy across all models (e.g., 5↔2 confusion reduced), and further improves convergence speed, after which ResNet18 reaches 97% accuracy in only 50 epochs.

=== 3.4 Loss Curves

We plot the training loss curves in first 100 epochs for selected models as follows:

#figure(image("loss_curve_CNN.png"), caption: [SimpleCNN])
#figure(image("loss_curve_ResNet.png"), caption: [ResNet18])
#figure(image("loss_curve_MobileNet.png"), caption: [MobileNetV2])
#figure(image("loss_curve_ShuffleNet.png"), caption: [ShuffleNetV2])
#figure(image("loss_curve_ViT.png"), caption: [ViT-Tiny])


== 4. Discussion and Conclusion

This work shows that FixMatch is highly effective for handwritten digit classification with extremely limited labels. Among all tested architectures, ResNet18 shows the best trade-off between capacity, speed, and final accuracy. SimpleCNN also performs competitively with much fewer parameters. Removing inappropriate augmentations like horizontal flip significantly boosts performance. Future work includes pretraining ViT or applying contrastive pretraining for stronger feature extractors.

#bibliography("refs.bib")
