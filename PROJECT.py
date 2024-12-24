import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('output_file.csv')

# Exclude 'Credit_History_Age' & 'Type_of_Loan' columns
df = df.drop(['Credit_History_Age', 'Type_of_Loan'], axis=1)

# 2. Explore each column
print(df.info())
print(df.describe())

# Clean up and pre-process data
# Handle missing values if any
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']):
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Normalize numerical values
scaler = StandardScaler()
df[df.select_dtypes(include=['float64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64']))

# Split data into train and test sets
X = df.drop('Credit_Mix', axis=1)
y = df['Credit_Mix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build optimal decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)


# Access overfitting concerns of each model
dt_model_pruned = DecisionTreeClassifier(max_depth=5)
dt_model_pruned.fit(X_train, y_train)
dt_train_predictions_pruned = dt_model_pruned.predict(X_train)
dt_test_predictions_pruned = dt_model_pruned.predict(X_test)
dt_train_accuracy_pruned = accuracy_score(y_train, dt_train_predictions_pruned)
dt_test_accuracy_pruned = accuracy_score(y_test, dt_test_predictions_pruned)
print("Pruned Decision Tree Training Accuracy:", dt_train_accuracy_pruned)
print("Pruned Decision Tree Test Accuracy:", dt_test_accuracy_pruned)

# Build neural network model to classify target value
nn_model = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', 
                         alpha=0.0001, batch_size='auto', learning_rate='constant', 
                         learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                         random_state=None, tol=0.0001, verbose=False, warm_start=False, 
                         momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                         validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                         n_iter_no_change=10, max_fun=15000)

nn_model.fit(X_train, y_train)
nn_train_predictions = nn_model.predict(X_train)
nn_test_predictions = nn_model.predict(X_test)
nn_train_accuracy = accuracy_score(y_train, nn_train_predictions)
nn_test_accuracy = accuracy_score(y_test, nn_test_predictions)
print("Neural Network Training Accuracy:", nn_train_accuracy)
print("Neural Network Test Accuracy:", nn_test_accuracy)

# Recommend a final model based on performance and overfitting concerns
# Based on accuracy scores and overfitting concerns, recommend a final model.
# For example:
if dt_test_accuracy_pruned > nn_test_accuracy:
    print("Final Model: Pruned Decision Tree")
else:
    print("Final Model: Neural Network")







