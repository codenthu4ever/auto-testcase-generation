import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample requirements and corresponding test cases
def generate_data(num_requirements):
    requirements = []
    test_cases = []

    for i in range(num_requirements):
        requirement = f"The BMS shall monitor parameter {i+1}."
        test_case = f"Verify that the BMS monitors parameter {i+1}."
        requirements.append(requirement)
        test_cases.append(test_case)

    return requirements, test_cases

# Assign labels to requirements based on predefined rules
def assign_labels(requirements):
    labels = []
    for requirement in requirements:
        if "state of charge" in requirement.lower():
            labels.append(0)
        elif "voltage" in requirement.lower() and "cell" in requirement.lower():
            labels.append(1)
        elif "cell balancing" in requirement.lower():
            labels.append(2)
        elif "fault detection" in requirement.lower():
            labels.append(3)
        else:
            labels.append(random.randint(0, 3))  # Assign a random label if requirement does not match predefined rules
    return labels


# Generate data
num_requirements = 100000
requirements, test_cases = generate_data(num_requirements)

# Assign labels to requirements
labels = assign_labels(requirements)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(requirements, labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(requirements, labels, test_size=0.2, random_state=42, stratify=labels)


# Create a pipeline with CountVectorizer and Multinomial Naive Bayes classifier
# model = make_pipeline(CountVectorizer(), MultinomialNB())
# model = make_pipeline(CountVectorizer(), MultinomialNB(class_prior=None, fit_prior=False, class_weight="balanced"))
# Calculate class priors based on class frequencies in the training data
class_counts = np.bincount(y_train)
class_priors = class_counts / len(y_train)

# Create a pipeline with CountVectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB(class_prior=class_priors))



# Train the model
model.fit(X_train, y_train)

# Predict test cases for new requirements
new_requirements = [
    "The BMS shall monitor state of charge (SoC) of the electric vehicle battery in real-time.",
    "The BMS shall monitor the voltage of each individual cell within the battery pack.",
    "The BMS shall perform cell balancing to ensure uniform charge levels across all cells in the battery pack.",
    "The BMS shall detect and diagnose faults related to battery cells, sensors, and communication interfaces."
]

predicted_labels = model.predict(new_requirements)

# Print predicted labels and test case names for new requirements
for requirement, label in zip(new_requirements, predicted_labels):
    print(f"Requirement: {requirement}")
    print(f"Predicted Test Case Label: Test Case {label + 1}")  # Print the predicted label
    print(f"Predicted Test Case: {test_cases[label]}")  # Access the corresponding test case name using the predicted label
    print()

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
