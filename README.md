# Pytorch-WaveNet

Pytorch implementation of DeepMind's WaveNet utilizing PyTorch

by: Jacqueline Abalo and John Donaghy

Dependencies: pytorch, torchvision, numpy, matplotlib, scipy

Dynamically built Wavenet model entirely described using command-line arguments.  Driven by train.py with model definitions in model.py and data handling in data.py.  
Data.py houses functionality to load LJSpeech dataset and VCTK dataset.  
Also made available is functionality to load Partita for Violin No. 2 by ohann Sebastian Bach, credit: Vincent Herrmann.
MuLawExpanding and MuLawDecoding in transforms.py credit: Sungwon Kim



Basic Run Example:

python train.py -dataset [dataset] --data_path [path/to/dataset/] --epochs [# epochs] --batch_size [# samples per batch] --use_cuda True

Further usage options available via: python train.py --help 
