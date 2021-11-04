# RUL-RVE

RUL-RVE is a new framework based on a novel recurrent version of a variational encoding for the assessment of engine health monitoring data in aircraft. The latent space learned by the model, trained with the historical data recorded by the sensors of these engines, is used to build a visual and self-explanatory map that can evaluate the rate of deterioration of the engines. High prognostic accuracy in estimating the RUL is achieved by introducing a penalty for estimation errors on a regression model built on top of the learned features of the encoder.

# Files in this Repository
- \data: samples with which to train the model.
- \images: folder to save images of the model latent space during training.
- \models: folder containing some trained models.
- RULRVE.ipynb: Jupyter notebook with the model implementation and results of a case study.
- experimentalResults.ipynb: Jupyter notebook to reproduce the saved models results.
- main.py: definition and model training.
- model.py: model architecture definition.
- utils.py: some helper functions.
