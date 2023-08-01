# Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms

Prediction of Heart diseases one can get in near future based on his previous traits.

The objective of 'HEARTPLUS Heart Disease Prediction System' is to identify individuals who are at high risk of developing heart disease using ML models. The goal is to develop accurate and reliable models that can aid in early detection and prevention of heart disease, ultimately leading to better patient outcomes and improved public health.

![image](https://github.com/ParthT1862/Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms/assets/100213897/f712d52d-d920-4df3-a0bd-c8b05a39d840)

# INTRODUCTION
Heart disease is one of the leading causes of death worldwide, and early detection of this condition is critical for preventing serious health complications. With the increasing availability of medical data and the advancement of ML algorithms, heart disease prediction has become a popular application of AI in healthcare. Heart disease prediction aims to identify individuals who are at a higher risk of developing cardiovascular disease using various clinical and demographic features, such as age, gender, blood pressure, cholesterol levels, and lifestyle factors. By leveraging the power of ML, doctors can accurately predict the likelihood of developing heart disease and take preventive measures to reduce the risk of heart attacks, strokes, and other cardiovascular complications. In this way, heart disease prediction models have the potential to save countless lives and improve the overall quality of healthcare.

# PROPOSED SYSTEM
Heart disease is a prevalent health condition that affects millions of people worldwide. It is a leading cause of death, and early detection and prevention can significantly improve patient outcomes. Therefore, a heart disease prediction system is essential to identify individuals who are at risk of developing heart disease. Our proposed heart disease prediction system is an advanced machine learning-based system designed to accurately predict the likelihood of heart disease in individuals. The system is built on a large dataset of clinical and demographic information of patients who have previously been diagnosed with heart disease.

![image](https://github.com/ParthT1862/Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms/assets/100213897/6b88c68b-5c4a-46c8-9de6-3cd6ffb8e602)

# MACHINE LEARNING MODELS USED
# 1. MULTILAYER PERCEPTRON
Neural networks are a type of ML algorithm that are capable of learning complex relationships between input features and output labels. Multilayer perceptron (MLP) is a type of artificial neural network that has been used in various fields including medical diagnosis. It’s also used in the diagnosis of heart disease. The MLP model consists of multiple layers of interconnected nodes that learn to represent the data and make predictions based on this representation. In the case of heart disease diagnosis, MLP can learn patterns in the data that are indicative of heart disease, such as high bp, abnormal heart rhythms, or high chol. levels. MLP shows promising results in diagnosing heart disease and can potentially be used as a tool to assist doctors in making accurate and timely diagnosis.

```python
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build neural network model
model2 = Sequential()
model2.add(Dense(32, input_dim=X_train.shape[1], activation="relu"))
model2.add(Dense(16, activation="relu"))
model2.add(Dense(1, activation="sigmoid"))

# Compile model
optimizer = Adam(learning_rate=0.001)
model2.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train model
history = model2.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Make predictions on testing set
y_pred2 = model2.predict(X_test)
y_pred2 = (y_pred2 > 0.5)
mlp_model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=1000)
```

# 2. XGBOOST
XGBoost is a popular ML algorithm that uses decision trees to create a predictive model. It is effective when dealing with large datasets with many features, and it is often used for classification. If we apply it to our dataset, we could predict whether or not a patient has heart disease based on their characteristics. It could be effective for our task because it is well-suited for handling complex relationships b/w features & outcomes, which is likely the case with heart disease. The specific outcome of it depends on a no. of factors, including specific implementation of the algorithm, the hyperparameters used, the quality of the data & the size & complexity of the dataset. However, we would hope to see an accurate predictive model that could be used to identify patients who are at risk of heart disease.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an XGBoost classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

#Predicted the y values using the model
y_pred = xgb_model.predict(X_test)
```

# 3. ADABOOST
AdaBoost is a popular ML algorithm used to improve the accuracy of predictions in binary classification problems. It works by iteratively training a sequence of weak classifiers and combining their outputs into a strong classifier. For heart disease prediction, AdaBoost can be used to identify patterns and features in medical data that are predictive of heart disease. By analyzing factors such as age, blood pressure, cholesterol levels, and family history, an AdaBoost-based heart disease prediction system can accurately assess the likelihood of a patient developing heart disease. This can enable early intervention and help prevent serious health complications associated with heart disease.

```python
# Define the AGABoost model
model3 = AdaBoostClassifier(
    n_estimators=1,
    learning_rate=0.3,
    random_state=1,
    base_estimator=RandomForestClassifier()
)

# Train the model
model3.fit(X_train, y_train)

# Evaluate the model
score = model3.score(X_test,y_test)

#Predicted y values
y_pred3=model3.predict(X_test)
```

# 4. LIGHTGBM
LightGBM is a gradient boosting model that is designed to be highly efficient and scalable, making it well-suited for large-scale ML tasks. It is often used in the context of heart disease prediction systems, where it can be used to analyze large volumes of medical data and identify patterns that indicate heart disease. LightGBM works by building decision trees in hierarchical manner where each tree is optimized to minimize loss function. By combining multiple decision trees in an ensemble, LightGBM can create a highly accurate model that is capable of accurately predicting likelihood of a patient developing heart disease. This is valuable for early detection and intervention for preventing serious health issues associated with heart disease.

```python
#Defined the parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'force_row_wise':'true'
}

#Train the model
lgb_model = lgb.train(params, train_data,100)

#Predict values using the model
y_pred4 = lgb_model.predict(X_test)

#Define the classifier
lgbm_model=lgb.LGBMClassifier()
```

# 5. WEIGHTED BAGGING
Weighted bagging is a ML technique that combines the power of bagging with the addition of weighted samples. Bagging is a technique where multiple models are trained on different subsamples of the same dataset to reduce the variance of the overall model. In weighted bagging, the samples are assigned different weights based on their importance. This enables the algo to pay more attention to the samples that are most informative for the heart disease prediction task. It can be used in heart disease prediction to improve the accuracy. By incorporating sample weighting, the algorithm can better identify the most informative features in medical data, such as age, gender, blood pressure, and chol. levels, that are most relevant to predicting heart disease.

```python
# Create the voting classifier
voting_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('mlp', mlp_model), ('ada', model3), ('lgbm', lgbm_model)],
    voting='hard',
    weights=(30, 10, 40, 20)
)

# Train the voting classifier on the training data
voting_model.fit(X_train, y_train)
```
# FIGURES AND GRAPHS
# 1) DATASET AFTER PREPROCESSING
![image](https://github.com/ParthT1862/Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms/assets/100213897/325a05cf-bf92-4a69-b444-66be8405f2bb)

# 2) CORRELATION HEATMAP
![image](https://github.com/ParthT1862/Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms/assets/100213897/20f559c2-b643-4455-8d96-6a1c6696abfd)

# RESULTS
![image](https://github.com/ParthT1862/Heart-Disease-Prediction-System-using-complex-Machine-Learning-Algorithms/assets/100213897/a1e146d2-bc99-47b4-bb05-16c3fcb382d4)
