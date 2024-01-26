# RUL-RVE

RUL-RVE is a framework based on a novel recurrent version of variational encoding for the assessment of engine health monitoring data in aircraft. The latent space learned by the model, trained with the historical data recorded by the sensors, is used to build a visual and self-explanatory map that can evaluate the rate of deterioration of the engines. High prognostic accuracy in estimating the RUL is achieved by introducing a penalty for estimation errors on a regression model built on top of the learned features of the encoder.

The paper is published in Reliability Engineering and System Safety and is publicly available at https://www.sciencedirect.com/science/article/pii/S0951832022000321?via%3Dihub.

You can find a Gradio demo of the model in https://huggingface.co/spaces/NahuelCosta/RUL-Variational

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/blob/main/RULRVE.ipynb)

[!IMPORTANT]
🚨 The code of this repository will no longer be mantained here. You can check [rapidae](https://github.com/NahuelCostaCortez/rapidae) where this model, along with other autoencoder models is up-to-date.

# Files in this Repository
- \data: samples with which to train the model.
- \images: folder to save images of the model latent space during training.
- \models: folder containing some trained models.
- RULRVE.ipynb: Jupyter notebook with the model implementation and results of a case study.
- experimentalResults.ipynb: Jupyter notebook to reproduce the saved models results.
- main.py: definition and model training.
- model.py: model architecture definition.
- utils.py: some helper functions.
- requirements.txt: requirements for the project.
- Dockerfile: instructions to build a Docker image with all the necessary dependencies.

To execute the python code we recommend setting up a new python environment with packages matching the requirements.txt file. It can be easily done with anaconda: conda create --name --file requirements.txt. Another alternative is to run exactly the same environment under which this project was made with Docker. A Dockerfile is provided, which contains the set of instructions for creating a container with all the necessary packages and dependencies. The fastest way to set it up is to clone the reposity, open Visual Studio Code, and from the command palette select "Remote-containers: Open folder in Container".
