# loan-default-prediction

The core dataset is the Home Equity (hmeq) dataset.

## Kind of Models Included:

- Decision Tree
- Random Forest
- XGBoost

## Repository Content:

- FastAPI endpoints that exposes each model
- Dockerized for distribution and scaling
- 3 tree-based models saved to container, load on inference time for predictions

## Model-side TODOs:

- Try more classifiers such as SVM and neural network based models
- Test if combining above models improves performance
- Can we gain similar / different insight from other data sources? Can we combine them?

## System-side TODOs:

- If there is a way to get / generate meaningful new data, add model re-train functionalities and backend storage