# A Hybrid Compact Neural Architecture for Visual Place Recognition

![](readme/hybrid.png)

In this release, we provide an open source implementation of the FlyNet supervised learning experiments
in [A Hybrid Compact Neural Architecture for Visual Place Recognition](https://arxiv.org/pdf/1910.06840.pdf),
as submitted to RA-L with ICRA 2020 option (https://arxiv.org/abs/1910.06840).

## Abstract

State-of-the-art algorithms for visual place recognition can be broadly split into two categories: computationally expensive deep-learning/image retrieval based techniques with minimal biological plausibility, and computationally cheap, biologically inspired models that yield poor performance in real-world environments. In this paper we present a new compact and high-performing system that bridges this divide for the first time. Our approach comprises two key components: FlyNet, a compact, sparse two-layer neural network inspired by fruit fly brain architectures, and a one-dimensional continuous attractor neural network (CANN). Our FlyNet+CANN network combines the compact pattern recognition capabilities of the FlyNet model with the powerful temporal filtering capabilities of an equally compact CANN, replicating entirely in a neural network implementation the functionality that yields high performance in algorithmic localization approaches like SeqSLAM. We evaluate our approach and compare it to three state-of-the-art methods on two benchmark real-world datasets with small viewpoint changes and extreme appearance variations including different times of day (afternoon to night) where it achieves an AUC performance of 87%, compared to 60% for Multi-Process Fusion, 46% for LoST-X and 1% for SeqSLAM, while being 6.5, 310, and 1.5 times faster respectively.

## Datasets

The dataset needed to run this code can be downloaded from
[here](https://drive.google.com/open?id=1xrHKrHYgSqrMk9-XeC1qIe8UYDmOsgfd), which is a small subset of the Nordland dataset. This code can easily adapted to run across other, much larger datasets.

## Use FlyNet

We provide a demo on a subset of the Nordland dataset.

After downloading the dataset from here [here](https://drive.google.com/open?id=1xrHKrHYgSqrMk9-XeC1qIe8UYDmOsgfd), extract it into the `dataset/` folder and run:

	python main.py


## Requirements

This code was tested on [PyTorch](https://pytorch.org/) v1.0 and Python 3.6.


## License

FlyNet itself is released under the MIT License (refer to the LICENSE file for details).


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

	@article{
		FlyNetMC19,
		author = {Chanc\'an, Marvin and Hernandez-Nunez, Luis and Narendra, Ajay and Barron, Andrew B. and Milford, Michael},
		title = {A Compact Neural Architecture for Visual Place Recognition},
		volume = {abs/1910.06840},
		year = {2019},
		url = {https://arxiv.org/abs/1910.06840},
		archivePrefix = {arXiv},
		eprint = {1910.06840}
	}
