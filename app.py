import pandas as pd
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv('breast_cancer_dataset.csv')

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.head())

# Encode the target column 'diagnosis': B (Benign) -> 0, M (Malignant) -> 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature selection: Select the top 10 features based on statistical significance
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
}

# Initialize the MLPClassifier
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Set up GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model
grid_search.fit(X_train_selected, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Train the ANN model with the best parameters
best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train_selected, y_train)

# Evaluate the model

y_pred = best_mlp.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(accuracy))
print(classification_report(y_test, y_pred))


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_selected)
X_test = scaler.transform(X_test_selected)


# Streamlit UI
st.title("Breast Cancer Prediction App")

# Display dataset
if st.checkbox('Show Dataset'):
    st.write(df.head())

# User input for prediction
st.sidebar.header("User Input Features")

input_features = {
    'radius_mean': st.sidebar.number_input('Radius Mean', min_value=0.0, value=17.99),
    'texture_mean': st.sidebar.number_input('Texture Mean', min_value=0.0, value=10.38),
    'perimeter_mean': st.sidebar.number_input('Perimeter Mean', min_value=0.0, value=122.8),
    'area_mean': st.sidebar.number_input('Area Mean', min_value=0.0, value=1001.0, max_value=1326.0, step=1.0),
    'smoothness_mean': st.sidebar.number_input('Smoothness Mean', min_value=0.0, value=0.1184),
    'compactness_mean': st.sidebar.number_input('Compactness Mean', min_value=0.0, value=0.2776),
    'concavity_mean': st.sidebar.number_input('Concavity Mean', min_value=0.0, value=0.3001),
    'concave points_mean': st.sidebar.number_input('Concave Points Mean', min_value=0.0, value=0.1471),
    'symmetry_mean': st.sidebar.number_input('Symmetry Mean', min_value=0.0, value=0.2419),
    'fractal_dimension_mean': st.sidebar.number_input('Fractal Dimension Mean', min_value=0.0, value=0.07871),
    # Add more inputs here as needed
}

input_df = pd.DataFrame([input_features])

# Scale the input features
input_df_scaled = scaler.transform(input_df)

# Make prediction
prediction = best_mlp.predict(input_df_scaled)
prediction_label = 'Malignant' if prediction[0] == 1 else 'Benign'

# Display prediction
st.write(f"Prediction: {prediction_label}")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")
