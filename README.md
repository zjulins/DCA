# DCA
Learning Dynamic Context Augmentation for Global Entity Linking
========

### Installation

- Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8

### Usage

#### Train
python main.py --method SL --order offset --device 0 --n_epochs 100 --isDynamic 0
    --method: training method, SL or RL
    --order: decision order, offset or size
    --isDynamic: data augmentation method, 0: coherence+DCA, 1: coherence, 2: local model
