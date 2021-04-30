# Pseudo-IoU-for-Anchor-Free-Object-Detection

This is the repo to host the code for **Pseudo-IoU** in the following paper: [Arxiv link](https://arxiv.org/abs/2104.14082)

By [Jiachen Li](https://chrisjuniorli.github.io/), [Bowen Cheng](https://bowenc0221.github.io/), [Rogerio Feris](https://mitibmwatsonailab.mit.edu/people/rogerio-feris/), [Jinjun Xiong](https://researcher.watson.ibm.com/researcher/view.php?person=us-jinjun), [Thomas S.Huang](http://ifp-uiuc.github.io/), [Wen-Mei Hwu](http://impact.crhc.illinois.edu/People/Hwu/hwu.aspx) and [Humphrey Shi](https://www.humphreyshi.com).

Our code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) will be released soon.

## Introduction

<p align="center">
  <img src="figs/fig1.pdf" width="40%">
</p>

Current anchor-free object detectors are quite simple and effective yet lack accurate label assignment methods, which limits their potential in competing with classic anchor-based models that are supported by welldesigned assignment methods based on the Intersectionover-Union (IoU) metric. In this paper, we present Pseudo Intersection-over-Union (Pseudo-IoU): a simple metric that brings more standardized and accurate assignment rule into anchor-free object detection frameworks without any additional computational cost or extra parameters for training and testing, making it possible to further improve anchor-free object detection by utilizing training samples of good quality under effective assignment rules that have been previously applied in anchor-based methods. By incorporating Pseudo-IoU metric into an end-toend single-stage anchor-free object detection framework, we observe consistent improvements in their performanceon general object detection benchmarks such as PASCAL VOC and MSCOCO. Our method (single-model and singlescale) also achieves comparable performance to other recent state-of-the-art anchor-free methods without bells and whistles.

## Citation

```bibtex
@article{li2021pseudoiou,
  title={Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection},
  author={Jiachen Li, Bowen Cheng, Rogerio Feris, Jinjun Xiong, Thomas S.Huang, Wen-Mei Hwu and Humphrey Shi},
  journal={arXiv preprint arXiv:2104.14082},
  year={2021}
}
```

