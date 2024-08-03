import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import LinearRegression

# Parameters
num_persons = 100
num_stories = 20

# Generate respondent abilities
abilities = np.random.normal(0, 1, num_persons)

# Generate item parameters
# Difficulty parameters for 2 thresholds per story (since we have a 3-point scale)
difficulties = [np.sort(np.random.normal(0, 1, 2)) for _ in range(num_stories)]
# Discrimination parameters for each story
discriminations = np.random.uniform(0.5, 2, num_stories)

# Function to calculate probabilities in the graded response model
def graded_response_model(ability, discrimination, thresholds):
    probs = [expit(discrimination * (ability - thresholds[k])) for k in range(len(thresholds))]
    probs = [0] + probs + [1]
    probs = np.diff(probs)
    return probs

# Simulate responses
responses = np.zeros((num_persons, num_stories))

for i in range(num_persons):
    for j in range(num_stories):
        probs = graded_response_model(abilities[i], discriminations[j], difficulties[j])
        # Ensure probabilities are valid (non-negative and sum to 1)
        if not np.all(probs >= 0) or not np.isclose(np.sum(probs), 1.0):
            probs = [0.33, 0.33, 0.34]  # Default to equal probabilities if invalid
        responses[i, j] = np.random.choice(np.arange(1, 4), p=probs)

# Convert to DataFrame
df = pd.DataFrame(responses, columns=[f'Story_{j+1}' for j in range(num_stories)])
df['Ability'] = abilities

# Display the first few rows of the dataset
print(df)

# Optionally, save the dataset to a CSV file
df.to_csv('synthetic_dataset.csv', index=False)

# Fit the GRM model to the data
# Assume we fit the model here using an appropriate library, e.g., scikit-learn

# Define a range of theta values
theta = np.linspace(-3, 3, 100)

# Function to calculate probabilities for a given item
def calculate_probabilities(ability, discrimination, thresholds):
    probabilities = []
    for theta_val in theta:
        probs = graded_response_model(theta_val, discrimination, thresholds)
        probabilities.append(probs)
    return np.array(probabilities)

# Choose an example item (e.g., the first item)
item_index = 0
item_params = {
    'discrimination': discriminations[item_index],
    'thresholds': difficulties[item_index]
}

# Calculate probabilities for the chosen item across theta values
probabilities = calculate_probabilities(abilities[0], item_params['discrimination'], item_params['thresholds'])

# Plot the category response curves
plt.figure(figsize=(10, 6))
for category in range(probabilities.shape[1]):
    plt.plot(theta, probabilities[:, category], label=f'Category {category + 1}')

plt.xlabel('Ability (Theta)')
plt.ylabel('Probability')
plt.title('Graded Response Model - Category Response Curves')
plt.legend()
plt.grid(True)
plt.show()