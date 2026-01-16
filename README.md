# Machine Learning Samples

This repository holds a collection of self-contained Jupyter notebooks that demonstrate core machine learning algorithms using scikit-learn. Each notebook generates or loads data, walks through the model pipeline, and explains every step with clear markdown guidance.

## Getting Started
- Install Python 3.10 or newer.
- Create and activate a virtual environment.
- Install the common requirements: `pip install numpy scikit-learn matplotlib`.
- Open any notebook in JupyterLab, VS Code, or another notebook interface and run the cells in order.

## Notebook Guide

### Regression
- Linear regression walkthrough: `Linear_Regression.ipynb`
- Logistic regression classification: `Logistic_Regression.ipynb`

### Probabilistic & Linear Models
- Gaussian Naive Bayes example: `Naive_Bayes.ipynb`
- Support vector classifier with RBF kernel: `Support_Vector_Machines.ipynb`
- K-nearest neighbors classifier: `K_Nearest_Neighbors.ipynb`

### Decision Trees & Forests
- Single decision tree classifier: `Decision_Trees.ipynb`
- Random forest ensemble: `Random_Forests.ipynb`

### Neural Networks
- Multilayer perceptron classifier: `Neural_Networks.ipynb`

### Ensembles
- Extra Trees baseline ensemble: `Ensembles.ipynb`
- Bagging with decision trees: `Ensembles_Bagging.ipynb`
- AdaBoost boosting demo: `Ensembles_Boosting.ipynb`
- Soft voting combination of heterogeneous models: `Ensembles_Voting.ipynb`
- Stacking meta-ensemble: `Ensembles_Stacking.ipynb`

### Unsupervised Learning
- K-means clustering on synthetic blobs: `K_Means.ipynb`
- Principal component analysis for dimensionality reduction: `Principal_Component_Analysis.ipynb`

## Tips
- All datasets are synthetic, so you can rerun cells safely; random states ensure reproducible results.
- Feel free to swap in your own data by replacing the dataset generation cell.
- Add visualizations (e.g., `matplotlib` plots) if you want deeper insight into model behavior.

Enjoy exploring the notebooks and adapting them to your own projects.