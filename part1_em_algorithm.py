from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_data, prepare_heights
from src.em import fit_em, expectation_step
from src.plotting import plot_mixture


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_FILE = PROJECT_ROOT / "figures" / "part1_em_graph.png"


def evaluate_classification(heights, mothers, children, means, variances, proportions):
    """Compare the fitted components with known mother/child labels."""
    actual_labels = np.concatenate([
        np.zeros(len(mothers), dtype=int),
        np.ones(len(children), dtype=int),
    ])
    responsibilities = expectation_step(heights, means, variances, proportions)
    predicted_components = np.argmax(responsibilities, axis=1)

    correct_normal = np.sum(
        (predicted_components == 0) == (actual_labels == 0)
    )
    correct_reversed = np.sum(
        (predicted_components == 1) == (actual_labels == 0)
    )

    if correct_normal >= correct_reversed:
        predicted_labels = predicted_components
        mapping = "Component 1 = Mother; Component 2 = Child"
    else:
        predicted_labels = 1 - predicted_components
        mapping = "Component 2 = Mother; Component 1 = Child"

    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (actual_labels, predicted_labels), 1)
    accuracy = np.mean(actual_labels == predicted_labels)
    return mapping, matrix, accuracy


def main():
    data = load_data()
    mothers, children, heights = prepare_heights(data)

    print("=" * 60)
    print("PART 1: EM ALGORITHM")
    print("=" * 60)
    print("Number of families:", len(mothers))
    print("Number of children:", len(children))
    print("Total height observations:", len(heights))

    initial_means = np.percentile(heights, [25, 75]).astype(float)
    initial_variance = np.var(heights)
    initial_variances = np.array([initial_variance, initial_variance])
    initial_proportions = np.array([0.5, 0.5])

    from src.em import calculate_log_likelihood
    print("\nINITIAL PARAMETERS")
    print("-" * 60)
    print("Means:", initial_means)
    print("Variances:", initial_variances)
    print("Mixing proportions:", initial_proportions)
    print("Log-likelihood:", round(calculate_log_likelihood(
        heights, initial_means, initial_variances, initial_proportions
    ), 4))

    result = fit_em(heights)
    means = result["means"]
    variances = result["variances"]
    proportions = result["proportions"]

    print("\nINITIAL STATE AND FIRST TWO EM ITERATIONS")
    print(pd.DataFrame(result["history"]).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))

    print("\nEM finished after", result["iterations"], "iterations.")
    print("FINAL PARAMETERS")
    print("Means:", means)
    print("Variances:", variances)
    print("Mixing proportions:", proportions)
    print("Final log-likelihood:", round(result["log_likelihood"], 4))

    test_height = 68.0
    test_responsibilities = expectation_step(
        np.array([test_height]), means, variances, proportions
    )[0]
    print(f"\nPOSTERIOR PROBABILITIES FOR HEIGHT {test_height}")
    print(f"Component 1: {test_responsibilities[0]:.4f}")
    print(f"Component 2: {test_responsibilities[1]:.4f}")
    print("Probability total:", round(test_responsibilities.sum(), 4))

    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    plot_mixture(heights, means, variances, proportions, test_height, FIGURE_FILE)

    print("\nWHY NOT SPLIT AT THE GLOBAL MEAN?")
    print("A global-mean split creates a hard boundary and ignores overlap between distributions.")
    print("EM calculates soft probabilities using both component means, variances and mixing proportions.")
    print("The two components are statistical groups; they are not automatically labels for mothers and children.")

    print("\n" + "=" * 60)
    print("MOTHER VS CHILD CLASSIFICATION EVALUATION")
    print("=" * 60)
    mapping, matrix, accuracy = evaluate_classification(
        heights, mothers, children, means, variances, proportions
    )
    print("Selected component mapping:", mapping)
    print("\nConfusion matrix:")
    print("Rows = actual group; columns = predicted group")
    print("                    Predicted Mother  Predicted Child")
    print(f"Actual Mother       {matrix[0, 0]:16d}  {matrix[0, 1]:14d}")
    print(f"Actual Child        {matrix[1, 0]:16d}  {matrix[1, 1]:14d}")
    print(f"\nCorrect predictions: {int(np.trace(matrix))}")
    print(f"Total observations: {len(heights)}")
    print(f"Classification accuracy: {accuracy * 100:.2f}%")

    print("\nINTERPRETATION")
    print("The confusion matrix shows how many mothers and children were assigned to each predicted group.")
    print("Accuracy is the percentage of all observations assigned to the correct group after choosing the better component mapping.")
    print("This is an evaluation on the same observations used to fit EM, not a test of performance on new data.")
    print("Because mother and child heights overlap and the children may be of different ages, height alone may not separate the groups well.")


if __name__ == "__main__":
    main()
