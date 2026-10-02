# GLU-ResNet-Density-Porosity

Code for the density and porosity prediction part of the paper

> Bhakta, S., Kant, R., Biswas, S., Pandit, S.A., Singh, R., Singh, K.H., Maurya, S.P. *Integrated Quantitative Reservoir Characterization using Hybrid Particle Swarm Optimization and Gated Linear Unit Residual Network for Acoustic Impedance Inversion and Petrophysical Property Prediction.* Submitted to Computers & Geosciences.

The notebooks predict **bulk density** and **porosity** from post-stack seismic amplitude and the acoustic impedance obtained from Hybrid Particle Swarm Optimization (HPSO) inversion. The model is a Gated Linear Unit Residual Network (GLU-ResNet / GRN): a Variable Selection Network followed by a stack of GLU-Residual blocks and a linear output layer (Section 3.3 of the paper). Hyperparameters are tuned with Optuna (TPE sampler).

## What is in the repo

| file | what it does |
|---|---|
| `01_feature_relevance_SHAP.ipynb` | seismic attributes + XGBoost + SHAP ranking and correlation heat map (Section 3.1.1, Fig. 4) |
| `02_density_prediction_GRN.ipynb` | density prediction with the GLU-Residual network (Section 3.3, 4.2.3) |
| `03_porosity_prediction_GRN.ipynb` | porosity prediction, same workflow as 02 |
| `seismic_attributes.py` | functions for the seismic attributes (complex trace, spectral frequency, Ormsby band-pass, amplitude integration ...) |
| `make_synthetic_data.py` | creates the small synthetic example data set in `example_data/` |
| `quick_test.py` | runs all three notebooks on the example data in quick mode (test) |
| `example_data/` | synthetic well trace (`.csv`) and a small synthetic section (`.npy`) |
| `requirements.txt` | python packages (versions we used) |

**Data:** the well logs and seismic data of the D1 field (Mumbai Offshore Basin) used in the paper are confidential and can not be shared. So the repo contains only a *synthetic* example with the same format. The numbers from the example are only for testing the code, they are not the results of the paper.

## Installation

Python 3.10 was used (3.9 - 3.12 should also work).

```
git clone https://github.com/soutik451/GLU-ResNet-Density-Porosity.git
cd GLU-ResNet-Density-Porosity
pip install -r requirements.txt
```

For GPU, install the CUDA build of PyTorch from https://pytorch.org first. A GPU is not needed, everything also runs on CPU (just slower).

## Quick test

```
python quick_test.py
```

This runs the three notebooks one after another on `example_data/synthetic_well_trace.csv` with `QUICK_TEST = True` (3 Optuna trials, max 150 epochs, 2 seeds). It takes about 2 min on GPU and a few minutes on a laptop CPU. At the end it prints the metric tables and `QUICK TEST PASSED`. Results are written to `outputs/`.

You can also open the notebooks in Jupyter (`jupyter notebook` or VS Code) and run all cells; by default they use the example data.

## Using your own data

The input is a `.csv` or `.xlsx` file for each well trace with these columns (names can be changed in the settings cell):

| column | meaning |
|---|---|
| `Seismic_Amplitude` | post-stack seismic trace at the well |
| `Inverted_Impedance_HPSO` | acoustic impedance from the HPSO inversion at the well |
| `Density` | bulk density log (g/cc), resampled to the seismic sampling |
| `Porosity` | porosity log (v/v), resampled to the seismic sampling |

Each trace has 201 samples (we used the 1700-2100 ms window at 2 ms). In the settings cell of 02 / 03 set

```python
DATA_FILES = ["path/to/your_well.csv"]
QUICK_TEST = False
```

and run all cells. The settings with `QUICK_TEST = False` are the ones used in the paper:

* random split 75 % train / 10 % validation / 15 % blind test (seed 42)
* attributes of the amplitude and impedance traces, correlation filter (|r| > 0.95), standard scaling, PCA keeping 99.5 % of the variance (all fitted on the training samples only)
* Optuna TPE, 40 trials, search space hidden_dim {128, 192, 256, 384, 512}, n_blocks 3-8, dropout 0.05-0.30, objective = validation R2, early stopping patience 80
* final training: AdamW (lr 1e-3, weight decay 1e-4), cosine annealing with warm restarts, batch 32, loss = Huber + 0.1(1 - PCC), mixup (alpha 0.2), stochastic weight averaging in the last 25 % of the epochs, early stopping patience 150, 7 networks with different seeds averaged

Note: GPU training is not fully deterministic, so the numbers can change a little from run to run even with fixed seeds.

To predict a seismic section, give two `.npy` files of shape (n_traces, 201) with the amplitude and HPSO impedance (last section of notebook 02 / 03, `SECTION_AMP` and `SECTION_IMP`).

## Outputs (in `outputs/density`, `outputs/porosity`, `outputs/shap`)

* `*_metrics.csv` : R2, PCC, RMSE, MAE, MAPE, bias, NRMSE, explained variance for train / validation / test / full trace (Table 1)
* `*_predictions.csv` : measured and predicted values for every sample
* `*_log_crossplot.png` : predicted vs measured log and cross-plot (Fig. 10)
* `*_nrmse_curve.png`, `*_nrmse_history_member1.csv` : training / validation NRMSE vs epoch (Fig. S1)
* `*_optuna_trials.csv`, `*_best_params.json` : hyperparameter search (Table S1)
* `*_weights.pt`, `*_preprocessing.joblib` : trained networks and scaler / PCA
* `*_section.npy`, `*_section.png` : predicted section (Fig. 11)
* `shap_ranking_*.csv`, `shap_*.png`, `correlation_*.png` : feature relevance (Fig. 4)

## Notes

* The attributes are computed with a nominal sample interval of 1 ms (`SAMPLE_RATE_HZ = 1000` in `seismic_attributes.py`), as in the paper. For our 2 ms data this means the frequencies in the attribute names are two times the real frequency (e.g. `Filter_135_140_145_150` is the 67.5-75 Hz band). Keep it at 1000 to get the same results as the paper.
* The acoustic impedance input comes from the HPSO (PSO + quasi-Newton) inversion described in Kant et al. (2024a), Phys. Chem. Earth 136, 103754, https://doi.org/10.1016/j.pce.2024.103754.
* `shap.TreeExplainer` does not work with xgboost >= 3, so notebook 01 takes the TreeSHAP values from xgboost directly (`pred_contribs=True`); the values are the same.

## License

MIT, see `LICENSE`.

## Contact

Soutik Bhakta, Department of Earth Sciences, IIT Bombay - bhaktasoutik@gmail.com
Corresponding author: Prof. K.H. Singh - kumar.h.singh@iitb.ac.in
