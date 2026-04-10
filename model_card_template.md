# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Developer: Jher545
Version 1.0.0
Model Type: Random Forest Classifier from SKlearn
Training framework: Python 3.10 with Sci-kit-learn, Pandas and NumPy
Hyperparamenters: Default settings and n_estimators=100 and random_state=42
## Intended Use
Primary: Model is designed for predicting whether a individuals annual income exceeds $50,000 based on demographic and employment data
Target: Audience is researchers and developers interested in data analysis, MLOPS pipeline deployment
## Training Data
Source: Census Bureau Income Dataset
Sizw: 80% of original dataset was used for the training(approximately26,000 rows)
Preprocessing:Categorical features were transformed using OneHotEncoder and the target labels were binarized with LabelBinarizer
## Evaluation Data
Source: 20% of the data from Census dataset (approximately 6,500 rows)
Slicing: Model used categorical slices to be evaluated
## Metrics
Model was evaluated using Precision, Recall,F1 score to balance between identifying higher earners and avoiding any false positives.
Precision: 0.7419
Recall: 0.3684
F1 Score: 0.6863
## Ethical Considerations
Data Bias: Dataset has sensitive attributes such as Native Country, Race and Sex. There is a risk that the model could inherit bias from hiatorical data
Mitigation: Model slicing analysis was used to address fairness. Reviewing the slice_output it is possible to identify the model underperforming for some demographics
## Caveats and Recommendations
Recency: Data is based on 1994 Census, not reflecting current factors such as inflation, job market.
Recommendation: The model should be retrained on more recent data to ensure predictions are relevent to current times.