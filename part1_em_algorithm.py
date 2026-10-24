
# ============================================================
# PART 1: PROBABILITY DISTRIBUTIONS AND EM ALGORITHM
# Dataset: Galton Family Heights
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# ------------------------------------------------------------

data = pd.read_csv("GaltonFamilies.csv")

# One mother's height per family
mothers = data.drop_duplicates(subset="family")["mother"].to_numpy()

# All recorded children's heights
children = data["childHeight"].to_numpy()

# Combine heights to fit a two-component Gaussian mixture
heights = np.concatenate([mothers, children]).astype(float)

print("=" * 60)
print("PART 1: EM ALGORITHM")
print("=" * 60)
print("Number of families:", len(mothers))
print("Number of children:", len(children))
print("Total height observations:", len(heights))


# ------------------------------------------------------------
# 2. GAUSSIAN DISTRIBUTION FUNCTIONS
# ------------------------------------------------------------

def gaussian_log_pdf(x, mean, variance):
    """Return the log of a Gaussian probability density."""
    variance = max(variance, 1e-6)

    return -0.5 * (
        np.log(2 * np.pi * variance)
        + ((x - mean) ** 2) / variance
    )


def calculate_log_likelihood(x, means, variances, proportions):
    """Calculate the total log-likelihood of the mixture."""
    log_component_1 = (
        np.log(proportions[0])
        + gaussian_log_pdf(x, means[0], variances[0])
    )

    log_component_2 = (
        np.log(proportions[1])
        + gaussian_log_pdf(x, means[1], variances[1])
    )

    # Stable calculation of log(exp(a) + exp(b))
    maximum = np.maximum(log_component_1, log_component_2)

    log_total = maximum + np.log(
        np.exp(log_component_1 - maximum)
        + np.exp(log_component_2 - maximum)
    )

    return np.sum(log_total)


# ------------------------------------------------------------
# 3. INITIALIZE THE TWO GAUSSIAN COMPONENTS
# ------------------------------------------------------------

K = 2

# Start the means at the 25th and 75th percentiles
means = np.percentile(heights, [25, 75]).astype(float)

# Start both components with the same variance and weight
initial_variance = np.var(heights)
variances = np.array([initial_variance, initial_variance])

proportions = np.array([0.5, 0.5])

print("\nINITIAL PARAMETERS")
print("-" * 60)
print(f"Means: {means}")
print(f"Variances: {variances}")
print(f"Mixing proportions: {proportions}")
print(
    "Log-likelihood:",
    round(
        calculate_log_likelihood(
            heights, means, variances, proportions
        ),
        4,
    ),
)


# ------------------------------------------------------------
# 4. EXPECTATION STEP (E-STEP)
# ------------------------------------------------------------

def expectation_step(x, means, variances, proportions):
    """Calculate each observation's probability for each component."""

    log_prob_1 = (
        np.log(proportions[0])
        + gaussian_log_pdf(x, means[0], variances[0])
    )

    log_prob_2 = (
        np.log(proportions[1])
        + gaussian_log_pdf(x, means[1], variances[1])
    )

    maximum = np.maximum(log_prob_1, log_prob_2)

    prob_1 = np.exp(log_prob_1 - maximum)
    prob_2 = np.exp(log_prob_2 - maximum)

    total = prob_1 + prob_2

    responsibility_1 = prob_1 / total
    responsibility_2 = prob_2 / total

    return np.column_stack(
        (responsibility_1, responsibility_2)
    )


# ------------------------------------------------------------
# 5. MAXIMIZATION STEP (M-STEP)
# ------------------------------------------------------------

def maximization_step(x, responsibilities):
    """Update the means, variances and mixing proportions."""

    effective_counts = responsibilities.sum(axis=0)

    new_means = (
        responsibilities.T @ x
    ) / effective_counts

    new_variances = np.array([
        np.sum(
            responsibilities[:, k]
            * (x - new_means[k]) ** 2
        ) / effective_counts[k]
        for k in range(K)
    ])

    new_variances = np.maximum(new_variances, 1e-6)

    new_proportions = effective_counts / len(x)

    return new_means, new_variances, new_proportions


# ------------------------------------------------------------
# 6. RUN EM AND TRACK THE INITIAL STATE AND ITERATIONS 1 AND 2
# ------------------------------------------------------------

history = []

def save_state(iteration, means, variances, proportions):
    history.append({
        "Iteration": iteration,
        "Mean 1": means[0],
        "Mean 2": means[1],
        "Variance 1": variances[0],
        "Variance 2": variances[1],
        "Mixing proportion 1": proportions[0],
        "Mixing proportion 2": proportions[1],
        "Log-likelihood": calculate_log_likelihood(
            heights, means, variances, proportions
        ),
    })


save_state(0, means, variances, proportions)

previous_log_likelihood = calculate_log_likelihood(
    heights, means, variances, proportions
)

for iteration in range(1, 1001):

    # E-step: calculate probabilities using current parameters
    responsibilities = expectation_step(
        heights, means, variances, proportions
    )

    # M-step: update parameters
    new_means, new_variances, new_proportions = (
        maximization_step(heights, responsibilities)
    )

    means = new_means
    variances = new_variances
    proportions = new_proportions

    current_log_likelihood = calculate_log_likelihood(
        heights, means, variances, proportions
    )

    if iteration <= 2:
        save_state(iteration, means, variances, proportions)

    # Stop when the log-likelihood changes very little
    if abs(current_log_likelihood - previous_log_likelihood) < 1e-6:
        break

    previous_log_likelihood = current_log_likelihood


# Print the required tracking table
tracking_table = pd.DataFrame(history)

print("\nINITIAL STATE AND FIRST TWO EM ITERATIONS")
print(tracking_table.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

print("\nEM finished after", iteration, "iterations.")
print("FINAL PARAMETERS")
print("Means:", means)
print("Variances:", variances)
print("Mixing proportions:", proportions)
print(
    "Final log-likelihood:",
    round(
        calculate_log_likelihood(
            heights, means, variances, proportions
        ),
        4,
    ),
)


# ------------------------------------------------------------
# 7. POSTERIOR PROBABILITIES FOR A TEST HEIGHT
# ------------------------------------------------------------

test_height = 68.0

test_responsibilities = expectation_step(
    np.array([test_height]), means, variances, proportions
)[0]

print(f"\nPOSTERIOR PROBABILITIES FOR HEIGHT {test_height}")
print(f"Component 1: {test_responsibilities[0]:.4f}")
print(f"Component 2: {test_responsibilities[1]:.4f}")
print(
    "Probability total:",
    round(test_responsibilities.sum(), 4),
)


# ------------------------------------------------------------
# 8. GRAPH THE DATA AND THE TWO GAUSSIAN COMPONENTS
# ------------------------------------------------------------

x_grid = np.linspace(heights.min() - 2, heights.max() + 2, 500)

def gaussian_pdf(x, mean, variance):
    return np.exp(gaussian_log_pdf(x, mean, variance))


component_1 = proportions[0] * gaussian_pdf(
    x_grid, means[0], variances[0]
)

component_2 = proportions[1] * gaussian_pdf(
    x_grid, means[1], variances[1]
)

mixture_density = component_1 + component_2

plt.figure(figsize=(10, 6))

plt.hist(
    heights,
    bins=25,
    density=True,
    alpha=0.5,
    label="Observed heights",
)

plt.plot(
    x_grid,
    component_1,
    linewidth=2,
    label="Gaussian component 1",
)

plt.plot(
    x_grid,
    component_2,
    linewidth=2,
    label="Gaussian component 2",
)

plt.plot(
    x_grid,
    mixture_density,
    linewidth=2.5,
    label="Combined mixture",
)

plt.axvline(
    test_height,
    linestyle="--",
    label=f"Test height = {test_height}",
)

plt.xlabel("Height (inches)")
plt.ylabel("Probability density")
plt.title("Two-Component Gaussian Mixture Fitted Using EM")
plt.legend()
plt.tight_layout()
plt.savefig("part1_em_graph.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 9. EXPLAIN WHY WE DO NOT SIMPLY SPLIT AT THE GLOBAL MEAN
# ------------------------------------------------------------

print("\nWHY NOT SPLIT AT THE GLOBAL MEAN?")
print(
    "A global-mean split creates a hard boundary and ignores "
    "overlap between distributions."
)
print(
    "EM calculates soft probabilities using both component "
    "means, variances and mixing proportions."
)
print(
    "The two components are statistical groups; they are not "
    "automatically labels for mothers and children."
)

# ------------------------------------------------------------
# 10. EVALUATE EM COMPONENTS AGAINST THE KNOWN GROUP LABELS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MOTHER VS CHILD CLASSIFICATION EVALUATION")
print("=" * 60)

# The heights array was created with mothers first and children second.
actual_labels = np.concatenate([
    np.zeros(len(mothers), dtype=int),  # 0 = Mother
    np.ones(len(children), dtype=int),   # 1 = Child
])

# Get the most likely component for every observation.
all_responsibilities = expectation_step(
    heights, means, variances, proportions
)
predicted_components = np.argmax(all_responsibilities, axis=1)

# EM component numbers do not automatically mean Mother or Child.
# Test both possible mappings and select the one with more correct labels.
correct_if_component_1_is_mother = np.sum(
    (predicted_components == 0) == (actual_labels == 0)
)

correct_if_component_2_is_mother = np.sum(
    (predicted_components == 1) == (actual_labels == 0)
)

if correct_if_component_1_is_mother >= correct_if_component_2_is_mother:
    # Component 1 = Mother; Component 2 = Child
    predicted_labels = predicted_components.copy()
    mapping = "Component 1 = Mother; Component 2 = Child"
else:
    # Component 2 = Mother; Component 1 = Child
    predicted_labels = 1 - predicted_components
    mapping = "Component 2 = Mother; Component 1 = Child"

# Calculate the confusion matrix.
# Rows = actual labels; columns = predicted labels.
# Label 0 = Mother, Label 1 = Child.
confusion_matrix = np.zeros((2, 2), dtype=int)

for actual, predicted in zip(actual_labels, predicted_labels):
    confusion_matrix[actual, predicted] += 1

true_mothers = confusion_matrix[0, 0]
mothers_predicted_as_children = confusion_matrix[0, 1]
children_predicted_as_mothers = confusion_matrix[1, 0]
true_children = confusion_matrix[1, 1]

accuracy = np.mean(actual_labels == predicted_labels)

print("Selected component mapping:", mapping)
print("\nConfusion matrix:")
print("Rows = actual group; columns = predicted group")
print("                    Predicted Mother  Predicted Child")
print(
    f"Actual Mother       {true_mothers:16d}"
    f"  {mothers_predicted_as_children:14d}"
)
print(
    f"Actual Child        {children_predicted_as_mothers:16d}"
    f"  {true_children:14d}"
)

print(f"\nCorrect predictions: {np.sum(actual_labels == predicted_labels)}")
print(f"Total observations: {len(actual_labels)}")
print(f"Classification accuracy: {accuracy * 100:.2f}%")

print("\nINTERPRETATION")
print(
    "The confusion matrix shows how many mothers and children "
    "were assigned to each predicted group."
)
print(
    "Accuracy is the percentage of all observations assigned "
    "to the correct group after choosing the better component mapping."
)
print(
    "This is an evaluation on the same observations used to fit EM, "
    "not a test of performance on new data."
)
print(
    "Because mother and child heights overlap and the children may "
    "be of different ages, height alone may not separate the groups well."
)
 