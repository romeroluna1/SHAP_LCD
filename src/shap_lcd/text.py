"""Local SHAP-LCD utilities for multiclass text classifiers.

This module implements the local quantities used by SHAP-LCD:

    Delta P = P(C_j | x) - P(C_r | x)
    Delta S_m = SHAP_m(C_j) - SHAP_m(C_r)

Positive Delta S values favour C_j and negative values favour C_r.
"""

from __future__ import annotations

import re
import html
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import numpy as np
import pandas as pd
import shap


PLOT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    "of", "to", "in", "on", "for", "from", "with", "by", "at", "as",
    "into", "over", "under", "after", "before", "about", "between",
    "through", "during", "against", "without", "within", "among",
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could", "may",
    "might", "must", "shall", "should", "it", "its", "this", "that",
    "these", "those", "he", "she", "they", "them", "his", "her",
    "their", "theirs", "we", "us", "our", "ours", "you", "your",
    "yours", "i", "me", "my", "mine", "not", "no", "said", "says",
    "say", "according", "lt", "gt", "br", "amp", "quot", "nbsp",
    "http", "https", "www", "com", "org", "net", "cm", "ct", "st",
    "ro", "ron", "ele", "ics", "en", "ny",
}


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name."""
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return value or "item"


class TextSHAPLCDExplainer:
    """Compute and visualise one local SHAP-LCD text explanation.

    Parameters
    ----------
    predict_proba:
        Callable receiving a list of strings and returning an array with
        shape ``(n_texts, n_classes)`` in probability scale.
    tokenizer:
        Hugging Face tokenizer used by the classifier. It is passed to
        ``shap.maskers.Text`` so SHAP uses the same tokenisation.
    class_names:
        Mapping ``class_index -> class_name`` or a sequence in class order.
    output_dir:
        Directory in which figures and CSV files are written.
    """

    def __init__(
        self,
        predict_proba: Callable[[Sequence[str]], np.ndarray],
        tokenizer,
        class_names: Mapping[int, str] | Sequence[str],
        output_dir: str | Path,
        extra_plot_stopwords: Sequence[str] | None = None,
    ) -> None:
        self.predict_proba = predict_proba
        self.tokenizer = tokenizer
        self.class_names = self._normalise_class_names(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_stopwords = set(PLOT_STOPWORDS)
        if extra_plot_stopwords is not None:
            self.plot_stopwords.update(
                str(word).strip().lower()
                for word in extra_plot_stopwords
                if str(word).strip()
            )

        self.masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(
            self.predict_proba,
            self.masker,
            output_names=[self.class_names[i] for i in range(len(self.class_names))],
        )

    @staticmethod
    def _normalise_class_names(
        class_names: Mapping[int, str] | Sequence[str],
    ) -> dict[int, str]:
        if isinstance(class_names, Mapping):
            result = {int(k): str(v) for k, v in class_names.items()}
        else:
            result = {i: str(v) for i, v in enumerate(class_names)}

        expected = list(range(len(result)))
        if sorted(result) != expected:
            raise ValueError(
                "class_names must use consecutive indices 0, 1, ..., K-1."
            )
        return result

    def _class_index(self, value: int | str) -> int:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if idx not in self.class_names:
                raise ValueError(f"Unknown class index: {idx}")
            return idx

        inverse = {name: idx for idx, name in self.class_names.items()}
        if value not in inverse:
            raise ValueError(f"Unknown class name: {value}")
        return inverse[value]

    def probabilities(self, text: str) -> np.ndarray:
        probs = np.asarray(self.predict_proba([text]), dtype=float)
        if probs.shape != (1, len(self.class_names)):
            raise ValueError(
                "predict_proba must return shape "
                f"(1, {len(self.class_names)}); received {probs.shape}."
            )
        if not np.all(np.isfinite(probs)):
            raise ValueError("The model returned non-finite probabilities.")
        if not np.isclose(probs[0].sum(), 1.0, atol=1e-5):
            raise ValueError("The model output is not in probability scale.")
        return probs[0]

    def rank_probability_pairs(self, text: str) -> pd.DataFrame:
        """Rank all unordered class pairs by absolute probability difference."""
        probs = self.probabilities(text)
        rows = []
        for j, r in combinations(range(len(self.class_names)), 2):
            signed = float(probs[j] - probs[r])
            rows.append(
                {
                    "class_j_index": j,
                    "class_j": self.class_names[j],
                    "class_r_index": r,
                    "class_r": self.class_names[r],
                    "probability_j": float(probs[j]),
                    "probability_r": float(probs[r]),
                    "delta_probability": signed,
                    "abs_delta_probability": abs(signed),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("abs_delta_probability", ascending=False)
            .reset_index(drop=True)
        )

    def plot_probability_differences(
        self,
        text: str,
        instance_label: str = "Instance",
        dpi: int = 300,
        filename: str = "probability_differences.png",
        top_k: int | None = None,
    ) -> pd.DataFrame:
        ranking = self.rank_probability_pairs(text)
        if top_k is None:
            plot_ranking = ranking
        else:
            if int(top_k) <= 0:
                raise ValueError("top_k must be a positive integer or None.")
            plot_ranking = ranking.head(int(top_k)).copy()

        labels = plot_ranking["class_j"] + " vs " + plot_ranking["class_r"]

        fig_height = max(4.5, 0.48 * len(plot_ranking) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        y = np.arange(len(plot_ranking))
        bars = ax.barh(
            y,
            plot_ranking["abs_delta_probability"],
            color="#4C72B0",
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"$|\Delta P|$")
        ax.set_title(f"Probability differences - {instance_label}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar, value in zip(bars, plot_ranking["abs_delta_probability"]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                fontsize=9,
            )
        upper = max(
            1.0,
            float(plot_ranking["abs_delta_probability"].max()) * 1.16,
        )
        ax.set_xlim(0.0, upper)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        ranking.to_csv(
            self.output_dir / "probability_pair_ranking.csv", index=False
        )
        return ranking

    def explain_pair(
        self,
        text: str,
        class_j: int | str,
        class_r: int | str,
        instance_label: str = "Instance",
        top_n: int = 20,
        max_evals: int | None = None,
        batch_size: int | str = "auto",
        dpi: int = 300,
    ) -> pd.DataFrame:
        """Explain one text for an ordered pair ``C_j`` and ``C_r``."""
        j = self._class_index(class_j)
        r = self._class_index(class_r)
        if j == r:
            raise ValueError("class_j and class_r must be different classes.")

        call_kwargs = {"batch_size": batch_size}
        if max_evals is not None:
            call_kwargs["max_evals"] = int(max_evals)
        explanation = self.explainer([text], **call_kwargs)

        values = np.asarray(explanation.values[0], dtype=float)
        tokens = np.asarray(explanation.data[0], dtype=object).astype(str)
        if values.ndim != 2 or values.shape[1] != len(self.class_names):
            raise ValueError(
                "Unexpected SHAP shape. Expected (tokens, classes), received "
                f"{values.shape}."
            )

        delta_s = values[:, j] - values[:, r]
        table = pd.DataFrame(
            {
                "token_position": np.arange(len(tokens), dtype=int),
                "token": tokens,
                "shap_class_j": values[:, j],
                "shap_class_r": values[:, r],
                "deltaS": delta_s,
                "abs_deltaS": np.abs(delta_s),
                "favours": np.where(
                    delta_s > 0,
                    self.class_names[j],
                    np.where(delta_s < 0, self.class_names[r], "Neutral"),
                ),
            }
        )

        # Keep every token in the CSV, but remove empty fragments from the plot.
        plot_table = table[table["token"].str.strip().ne("")].copy()
        plot_table = plot_table.nlargest(min(top_n, len(plot_table)), "abs_deltaS")
        plot_table = plot_table.sort_values("deltaS")

        pair_tag = (
            f"{_safe_name(self.class_names[j])}_vs_"
            f"{_safe_name(self.class_names[r])}"
        )
        table.to_csv(
            self.output_dir / f"local_deltaS_{pair_tag}.csv", index=False
        )

        probs = self.probabilities(text)
        delta_p = float(probs[j] - probs[r])
        base_values = np.asarray(explanation.base_values[0], dtype=float)
        reconstructed_delta_p = float(
            (base_values[j] - base_values[r]) + delta_s.sum()
        )
        reconstruction_error = abs(delta_p - reconstructed_delta_p)

        # Reconstruct readable lexical units and aggregate repeated occurrences.
        # Numeric fragments, punctuation and one-character subword remnants are
        # omitted from the compact local-text representation.
        lexical_rows = []
        for row in table.itertuples():
            cleaned = self._clean_token(row.token).strip().lower()
            words = [
                word for word in re.findall(r"[a-z][a-z-]+", cleaned)
                if len(word) >= 3 and word not in self.plot_stopwords
            ]
            if not words:
                continue
            share = float(row.deltaS) / len(words)
            lexical_rows.extend((word, share) for word in words)

        if lexical_rows:
            cloud = pd.DataFrame(lexical_rows, columns=["word", "deltaS"])
            cloud = cloud.groupby("word", as_index=False)["deltaS"].sum()
            cloud["abs_deltaS"] = cloud["deltaS"].abs()
            cloud = cloud.nlargest(min(top_n, len(cloud)), "abs_deltaS")
            cloud = cloud.sort_values("abs_deltaS", ascending=False)
        else:
            cloud = pd.DataFrame(columns=["word", "deltaS", "abs_deltaS"])

        cloud.to_csv(
            self.output_dir / f"local_word_deltaS_{pair_tag}.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.94,
                rf"SHAP Differences $\Delta S$ ({self.class_names[j]}/{self.class_names[r]})",
                ha="center", va="center", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.885,
                f"{instance_label} | P({self.class_names[j]})={probs[j]:.3f}, "
                f"P({self.class_names[r]})={probs[r]:.3f}, "
                rf"$\Delta P$={delta_p:.3f}",
                ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.83,
                f"Magenta: {self.class_names[j]} | Green: {self.class_names[r]} | "
                rf"Size: $|\mathrm{{sum}}\;\Delta S|$",
                ha="center", va="center", fontsize=11)
        ax.plot([0.03, 0.97], [0.79, 0.79], color="#d0d0d0", linewidth=0.8)

        if len(cloud):
            minimum = float(cloud["abs_deltaS"].min())
            maximum = float(cloud["abs_deltaS"].max())
            span = maximum - minimum
            x, y = 0.03, 0.67
            renderer = fig.canvas.get_renderer()
            for row in cloud.itertuples():
                strength = 1.0 if span <= 1e-15 else (
                    (float(row.abs_deltaS) - minimum) / span
                )
                fontsize = 12.0 + 20.0 * strength
                token = str(row.word)
                weight = "bold" if strength >= 0.72 else "normal"
                colour = "#d81b60" if float(row.deltaS) > 0 else "#009688"
                artist = ax.text(
                    x, y, token, transform=ax.transAxes,
                    fontsize=fontsize, color=colour,
                    ha="left", va="center", fontweight=weight,
                )
                fig.canvas.draw()
                bbox = artist.get_window_extent(renderer=fig.canvas.get_renderer())
                right = ax.transAxes.inverted().transform((bbox.x1, bbox.y1))[0]

                # If the rendered word exceeds the usable width, move the
                # complete word to the following row and measure it again.
                if right > 0.97 and x > 0.03:
                    artist.remove()
                    x = 0.03
                    y -= 0.16
                    artist = ax.text(
                        x, y, token, transform=ax.transAxes,
                        fontsize=fontsize, color=colour,
                        ha="left", va="center", fontweight=weight,
                    )
                    fig.canvas.draw()
                    bbox = artist.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    )
                    right = ax.transAxes.inverted().transform(
                        (bbox.x1, bbox.y1)
                    )[0]

                x = right + 0.018

        fig.savefig(self.output_dir / f"deltaS_{pair_tag}.png",
                    dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        metadata = pd.DataFrame(
            [
                {
                    "instance_label": instance_label,
                    "class_j_index": j,
                    "class_j": self.class_names[j],
                    "class_r_index": r,
                    "class_r": self.class_names[r],
                    "probability_j": float(probs[j]),
                    "probability_r": float(probs[r]),
                    "delta_probability": delta_p,
                    "reconstructed_delta_probability": reconstructed_delta_p,
                    "reconstruction_error": reconstruction_error,
                    "n_shap_tokens": len(tokens),
                    "text": text,
                }
            ]
        )
        metadata.to_csv(
            self.output_dir / f"local_metadata_{pair_tag}.csv", index=False
        )

        print("\nLocal SHAP-LCD explanation")
        print(f"Pair              : {self.class_names[j]} vs {self.class_names[r]}")
        print(f"Delta P           : {delta_p:.8f}")
        print(f"Reconstructed     : {reconstructed_delta_p:.8f}")
        print(f"Reconstruction err: {reconstruction_error:.3e}")
        print(f"Results directory : {self.output_dir}")
        return table

    @staticmethod
    def _clean_token(token: str) -> str:
        token = str(token)
        token = token.replace("Ä ", " ")
        token = token.replace("â–", " ")
        token = token.replace("##", "")
        return token

    @staticmethod
    def _colour_span(token: str, value: float, max_abs: float) -> str:
        strength = 0.0 if max_abs <= 0 else min(abs(value) / max_abs, 1.0)
        alpha = 0.15 + 0.70 * strength
        if value > 0:
            colour = f"rgba(214, 39, 180, {alpha:.3f})"
        elif value < 0:
            colour = f"rgba(44, 160, 44, {alpha:.3f})"
        else:
            colour = "rgba(255, 255, 255, 0.0)"
        return (
            f'<span style="background-color:{colour}; padding:2px 3px; '
            f'margin:1px; border-radius:3px;">{html.escape(token)}</span>'
        )

    def save_text_highlight(
        self,
        text: str,
        token_table: pd.DataFrame,
        class_j: int | str,
        class_r: int | str,
        instance_label: str = "Instance",
        filename: str | None = None,
    ) -> Path:
        """Save an HTML view in which token colour encodes Delta S sign."""
        j = self._class_index(class_j)
        r = self._class_index(class_r)
        max_abs = float(token_table["abs_deltaS"].max()) if len(token_table) else 0.0
        spans = []
        for row in token_table.sort_values("token_position").itertuples():
            token = self._clean_token(row.token)
            if token.strip():
                spans.append(self._colour_span(token, float(row.deltaS), max_abs))

        probs = self.probabilities(text)
        pair_tag = (
            f"{_safe_name(self.class_names[j])}_vs_"
            f"{_safe_name(self.class_names[r])}"
        )
        if filename is None:
            filename = f"shap_lcd_text_{pair_tag}.html"

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SHAP-LCD text explanation</title>
</head>
<body style="font-family:Arial,sans-serif;line-height:1.8;max-width:1100px;">
  <h2>SHAP-LCD text explanation: {html.escape(self.class_names[j])} vs {html.escape(self.class_names[r])}</h2>
  <h3>{html.escape(instance_label)}</h3>
  <p><b>Predicted probabilities:</b>
     {html.escape(self.class_names[j])}: {probs[j]:.4f} |
     {html.escape(self.class_names[r])}: {probs[r]:.4f}</p>
  <p>
    <span style="background-color:rgba(214,39,180,.75);padding:3px 6px;border-radius:3px;">Magenta</span>
    favours <b>{html.escape(self.class_names[j])}</b>&nbsp;&nbsp;
    <span style="background-color:rgba(44,160,44,.75);padding:3px 6px;border-radius:3px;">Green</span>
    favours <b>{html.escape(self.class_names[r])}</b>
  </p>
  <div style="font-size:18px;border:1px solid #ddd;padding:14px;border-radius:6px;">
    {' '.join(spans)}
  </div>
</body>
</html>
"""
        path = self.output_dir / filename
        path.write_text(document, encoding="utf-8")
        return path
