"""Style commun des figures du rapport."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from _commun import FIGURES  # noqa: E402

BLEU, ORANGE, VERT, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
ENCRE, ENCRE2, GRILLE, AXE = "#1f1f1f", "#5f5f5a", "#e1e0d9", "#c3c2b7"
LARGEUR = 6.69   # largeur du texte du rapport (pouces)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXE, "axes.labelcolor": ENCRE2, "xtick.color": ENCRE2, "ytick.color": ENCRE2,
    "text.color": ENCRE, "axes.titlecolor": ENCRE, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42,
})


def virgule(x, decimales=0):
    return f"{x:.{decimales}f}".replace(".", ",").replace("-", "−")


def axe_virgule(axe, decimales):
    axe.set_major_formatter(FuncFormatter(lambda v, _: virgule(v, decimales)))


def enregistrer(fig, nom):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{nom}.{ext}", dpi=220)
    plt.close(fig)
    print("Figure écrite :", FIGURES / f"{nom}.pdf")
