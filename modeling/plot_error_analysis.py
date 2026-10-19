"""Create the two error-analysis graphs without training any models.

Run: python modeling/plot_error_analysis.py
Requires: matplotlib

Results are from the saved CatBoost income-ratio experiment, evaluated on
36,902 blackbox validation applicants. Values come from that run's report.md;
the comparison CSV was later rounded. The final holdout was not used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save images without requiring a GUI or notebook.
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "figures"
BLUE = "#2864A0"
ORANGE = "#D57C28"

# Original experiment: catboost_income_ratios
MODEL_NAMES = ["Original features", "With income ratios"]
AUC_SCORES = [0.783802, 0.784274]

# Both thresholds below apply to the original-feature model.
# The second threshold was selected on eyeball data by maximizing F1.
THRESHOLD_NAMES = ["0.500", "0.154 (selected on review data)"]
DEFAULTS_CAUGHT = [145, 1344]
FALSE_ALARMS = [103, 3661]


def save_chart(fig, name):
    """Save a slide image and a PDF, then release the figure."""
    for extension in ("png", "pdf"):
        filename = OUTPUT / f"{name}.{extension}"
        fig.savefig(filename, dpi=220, facecolor="white")
        print(f"Saved: {filename}")
    plt.close(fig)


def plot_model_comparison():
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = [1, 0]

    ax.scatter(AUC_SCORES, positions, s=130, color=[BLUE, ORANGE], zorder=3)
    for score, position in zip(AUC_SCORES, positions):
        ax.annotate(f"{score:.5f}", (score, position), xytext=(12, 0),
                    textcoords="offset points", va="center", fontweight="bold")

    ax.set_yticks(positions, MODEL_NAMES)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("ROC-AUC (higher is better)")
    ax.set_title("Adding income ratios did not clearly improve CatBoost", pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.supxlabel(
        "AUC change: +0.00047; 95% bootstrap interval: −0.00058 to +0.00172.\n"
        "The interval includes zero, so keep the original features.",
        fontsize=10,
    )
    save_chart(fig, "income_ratio_comparison")


def plot_threshold_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    positions = [0, 1]
    width = 0.32

    caught = ax.bar([x - width / 2 for x in positions], DEFAULTS_CAUGHT,
                    width, label="Defaults caught", color=BLUE)
    alarms = ax.bar([x + width / 2 for x in positions], FALSE_ALARMS,
                    width, label="False alarms", color=ORANGE)
    for bars in (caught, alarms):
        labels = [f"{int(bar.get_height()):,}" for bar in bars]
        ax.bar_label(bars, labels=labels, padding=5)

    ax.set_xticks(positions, THRESHOLD_NAMES)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Applicants")
    ax.set_ylim(0, 4300)
    ax.set_title("Lowering the threshold catches more defaults and raises false alarms", pad=15)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.supxlabel(
        "Original-feature CatBoost: recall rises from 4.9% to 45.1%.\n"
        "Same 36,902 validation applicants; final holdout untouched.",
        fontsize=10,
    )
    save_chart(fig, "threshold_tradeoff")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    plot_model_comparison()
    plot_threshold_tradeoff()


if __name__ == "__main__":
    main()
