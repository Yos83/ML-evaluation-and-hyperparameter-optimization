# SVM-K-NN-DT-RF-Model Applications

Evaluating Model Performance and Hyperparameter Optimization for Wine Quality Classification: SVM, K-NN, Decision Tree, and Random Forest Approaches

The purpose of this project is to selected the model with highest accuracy and model performance to classify Wines by quality. The comparison accross models is achieved by checking multiple Kernel types in Support Vector Machine (SVM), n_neighbors in K-nearest neighbours (K-NN) , maximum depth in Decision Tree (DT) and GridSearchCV hyperparameters tuning for Random Forest (RF) classification algorithms for Binary and Multi-class tasks.

Model performance metrics includes F1, Precision, Recall and Confusion Matrix to assess which model result in the highest accuracy.

Data set description: Two datasets (quality for red and white wines) related to red and white variants of Portuguese wine are investigated. For more details consult: http://www.vinhoverde.pt/en/. Available at: https://archive.ics.uci.edu/ml/datasets/wine+quality


### Summary of Model Performance and Recommendations:

- SVM with RBF Kernel (gamma=0.1) for Binary Classification recorded a high accuracy (76%) with strong F1 score (0.770), precision (0.762), and recall (0.761). The confusion matrix shows a high number of true negatives (135), indicating the model is effective at identifying low-quality wines. Challenges: The model struggles with data imbalance, leading to more false negatives (FN) than false positives (FP). It is better at predicting low-quality wines than high-quality ones.
Recommendations: Improve the model by addressing data imbalance, testing different models, or adjusting hyperparameters. Ensemble methods like Random Forest could be more effective.
Random Forest (RF) for Binary Classification:

- RF with n_estimators=70 and max_depth=14 achieved the highest accuracy wht 93.5% and strong metrics across F1, precision, and recall. Adjusting n_estimators and max_depth helped reduce overfitting.
Feature Importance: Alcohol, volatile acidity, and sulphates are key indicators of wine quality, with good reviews associated with higher alcohol and sulphates and lower volatile acidity.
Recommendations: RF is the best-performing model for classifying red wines. Further improvement can be achieved by fine-tuning hyperparameters.
SVM with RBF Kernel for Multi-Class Classification:

- The RBF kernel yielded the highest accuracy (64.5%) and outperformed the linear kernel across F1, recall, and precision. However, the confusion matrix shows high misclassification rates for minority classes.
Recommendations: Consider addressing class imbalance and model configuration to improve the performance further.
K-Nearest Neighbors (KNN) for Binary and Multi-Class Classification:

- KNN with n_neighbors=8 provided a good balance between accuracy (71.5% for binary classification) and complexity, avoiding overfitting better than larger values of n_neighbors.
Challenges: High false positive rates and poor AUC scores indicate the model struggles with minority classes, particularly for binary classification.
Recommendations: n_neighbors=8 is a better choice for KNN to balance performance and complexity. Addressing class imbalance and refining the model configuration could further improve results.
Decision Tree (DT) for Binary and Multi-Class Classification:

- DT with max_depth=None achieved strong metrics for both binary and multi-class tasks, with accuracy up to 72% for binary classification. However, high false positive rates for minority classes indicate potential issues.
Recommendations: While DT performs well, improving class balance, reducing noise, and fine-tuning the model could enhance its performance.

Overall, Random Forest and SVM with RBF Kernel are the top-performing models, with Random Forest being particularly robust. Addressing data imbalance and fine-tuning models are key to further improvements.

### Python libraries used in this project:
name: my_environment
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - glob2
  - os
  - matplotlib
  - seaborn
  - scipy
  - scikit-learn
  - imbalanced-learn
  - mlxtend
  - pip
  - pip:
    - stringcase
   
  ple
