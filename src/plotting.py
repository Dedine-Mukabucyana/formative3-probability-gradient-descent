import numpy as np
import matplotlib.pyplot as plt

from src.em import gaussian_pdf


def plot_mixture(heights, means, variances, proportions, test_height, output_path):
    """Plot observations and the fitted Gaussian mixture."""
    x_grid = np.linspace(heights.min() - 2, heights.max() + 2, 500)

    component_1 = proportions[0] * gaussian_pdf(
        x_grid, means[0], variances[0]
    )
    component_2 = proportions[1] * gaussian_pdf(
        x_grid, means[1], variances[1]
    )
    mixture_density = component_1 + component_2

    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=25, density=True, alpha=0.5,
             label="Observed heights")
    plt.plot(x_grid, component_1, linewidth=2,
             label="Gaussian component 1")
    plt.plot(x_grid, component_2, linewidth=2,
             label="Gaussian component 2")
    plt.plot(x_grid, mixture_density, linewidth=2.5,
             label="Combined mixture")
    plt.axvline(test_height, linestyle="--",
                label=f"Test height = {test_height}")
    plt.xlabel("Height (inches)")
    plt.ylabel("Probability density")
    plt.title("Two-Component Gaussian Mixture Fitted Using EM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
