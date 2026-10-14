"""
Classification metrics: accuracy, per-class precision / recall / F1, and a confusion-matrix plot.

Why more than accuracy? The surnames data is very imbalanced (about 47 % of
names are Russian). A model that answered "Russian" for everything would
already score ~47 % accuracy while being useless for the other 17 languages.
Per-class F1 and *macro* F1 (the unweighted mean over classes) weight every
language equally and expose that kind of failure.

    precision(c) = correct predictions of c / all predictions of c
    recall(c)    = correct predictions of c / all true examples of c
    F1(c)        = harmonic mean of precision and recall
"""


def confusion_matrix(y_true, y_pred, n_classes):
    """Count (true class, predicted class) pairs.

    Args:
        y_true: true class indices.
        y_pred: predicted class indices, same length.
        n_classes: number of classes.

    Returns:
        n_classes x n_classes nested list; [i][j] = examples of class i predicted as j.
    """
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t][p] += 1
    return cm


def _safe_div(a, b):
    """a / b, or 0.0 when b is 0 (e.g. precision of a class that is never predicted)."""
    return a / b if b else 0.0


def classification_report(y_true, y_pred, labels):
    """Accuracy, per-class precision/recall/F1/support, macro and weighted F1.

    Args:
        y_true: true class indices.
        y_pred: predicted class indices.
        labels: class names; labels[i] names class index i.

    Returns:
        {"accuracy", "macro_f1", "weighted_f1",
         "per_class": {label: {"precision", "recall", "f1", "support"}},
         "confusion_matrix": counts, rows = true class, columns = predicted}.
    """
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])                       # true examples of class i (row sum)
        predicted = sum(cm[r][i] for r in range(n))  # predictions of class i (column sum)
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    total = len(y_true)
    return {
        "accuracy": _safe_div(sum(cm[i][i] for i in range(n)), total),
        "macro_f1": _safe_div(sum(c["f1"] for c in per_class.values()), n),
        "weighted_f1": _safe_div(sum(c["f1"] * c["support"] for c in per_class.values()), total),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, labels, path, title="Confusion matrix (row-normalised)"):
    """Save a row-normalised confusion-matrix heatmap as a PNG.

    Rows are normalised so each shows how one true language's names were
    classified; the diagonal is then the per-class recall. Without this, the
    Russian row would dominate the colour scale.

    Args:
        cm: count matrix from confusion_matrix().
        labels: class names in index order.
        path: output PNG path.
        title: figure title.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display needed (CI, servers)
    import matplotlib.pyplot as plt

    norm = [[_safe_div(v, sum(row)) for v in row] for row in cm]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    ax.set_title(title)
    for i, row in enumerate(norm):
        for j, v in enumerate(row):
            if v >= 0.005:  # skip empty cells to keep the plot readable
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
