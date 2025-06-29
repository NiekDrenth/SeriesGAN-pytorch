# SeriesGAN implementation in pytorch

Adapts the original [SeriesGAN](https://github.com/samresume/SeriesGAN) code using pytorch.
Also significantly speeds up the training process.

## Instructions for training the model

If you have [uv](https://github.com/astral-sh/uv) installed (recommended):

``` sh
git clone https://github.com/NiekDrenth/SeriesGAN-pytorch.git
cd SeriesGAN-pytorch
uv pip install -r requirements.txt

```

Without uv:


``` sh
git clone https://github.com/NiekDrenth/SeriesGAN-pytorch.git
cd SeriesGAN-pytorch
```

Make sure you have python 3.13
Create a virtual environment
``` sh
python3.13 -m venv .venv
```

activate it

Mac/Linux:

``` sh
source .venv/bin/activate
```

Windows:

``` powershell
.\venv\Scripts\Activate.ps1
```

Install requirements and run:

``` sh
pip install -r requirements.txt
python train.py
```

## How to Cite
Please Cite the original authors of the paper.
The paper associated with this repository has been accepted at BigData 2024 as a regular paper for oral presentation. We kindly ask you to provide a citation to acknowledge our work.

Available on arXiv:
[https://arxiv.org/abs/2410.21203](https://arxiv.org/abs/2410.21203)

Here is the BibTeX citation for your reference:

 ```
@misc{eskandarinasab2024seriesgan,
      title={SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning}, 
      author={MohammadReza EskandariNasab and Shah Muhammad Hamdi and Soukaina Filali Boubrahimi},
      year={2024},
      eprint={2410.21203},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.21203}, 
}
