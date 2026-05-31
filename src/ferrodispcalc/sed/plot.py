from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .calc import DipoleSedResult


__all__ = ["plot_sed"]

_COMPONENTS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "total": 3,
}


def plot_sed(
    result: DipoleSedResult | np.ndarray,
    freq_THz: np.ndarray | None = None,
    qpoints: np.ndarray | None = None,
    q_distances: np.ndarray | None = None,
    component: str = "total",
    ax: plt.Axes | None = None,
    q_labels: tuple[str, str] = (r"$\Gamma$", "X"),
    freq_max: float | None = None,
    freq_interval: float = 5.0,
    cmap: str = "RdBu_r",
    colorbar_min: float | None = None,
    colorbar_max: float | None = None,
    use_contourf: bool = False,
    savepath: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one SED component.

    Parameters
    ----------
    result : DipoleSedResult | np.ndarray
        A :class:`DipoleSedResult`, or a raw SED array with shape
        ``(nfreq, nq, 4)`` or ``(nfreq, nq)``. If an array is provided,
        ``freq_THz`` and ``qpoints`` must also be provided.
    freq_THz : np.ndarray | None, optional
        Frequency axis in THz. Ignored when ``result`` is a
        :class:`DipoleSedResult`.
    qpoints : np.ndarray | None, optional
        Reduced q-points with shape ``(nq, 3)``. Ignored when ``result`` is a
        :class:`DipoleSedResult`.
    q_distances : np.ndarray | None, optional
        Cumulative q-path distances. If not provided, they are reconstructed
        from neighboring q-point differences.
    component : str, optional
        One of ``"x"``, ``"y"``, ``"z"``, or ``"total"``. Defaults to
        ``"total"``.
    ax : matplotlib.axes.Axes | None, optional
        Existing axes. A new figure and axes are created when ``None``.
    q_labels : tuple[str, str], optional
        Labels for the first and last q-points. Defaults to Gamma-X labels.
    freq_max : float | None, optional
        Maximum plotted frequency in THz. Defaults to the maximum available
        frequency.
    freq_interval : float, optional
        Frequency tick interval in THz. Defaults to ``5.0``.
    cmap : str, optional
        Matplotlib colormap. Defaults to ``"RdBu_r"``.
    colorbar_min, colorbar_max : float | None, optional
        Colorbar limits on the log-intensity scale.
    use_contourf : bool, optional
        Use ``contourf`` instead of the default ``imshow`` rendering.
    savepath : str | Path | None, optional
        Save the figure when provided.
    show : bool, optional
        Show the figure interactively before returning.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    sed, freq_THz, qpoints = _unpack_sed_input(result, freq_THz, qpoints)
    sed_2d = _select_component(sed, component)
    freq_THz = np.asarray(freq_THz, dtype=np.float64)
    qpoints = np.asarray(qpoints, dtype=np.float64)
    q_axis = _q_axis(qpoints, q_distances)

    if sed_2d.shape != (len(freq_THz), len(qpoints)):
        raise ValueError(
            "Selected SED component must have shape (nfreq, nq). "
            f"Got {sed_2d.shape}, expected {(len(freq_THz), len(qpoints))}."
        )

    sed_log = np.log(np.clip(sed_2d, np.finfo(float).tiny, None))
    vmin = np.trunc(sed_log.min()) if colorbar_min is None else colorbar_min
    vmax = np.trunc(sed_log.max()) if colorbar_max is None else colorbar_max
    if vmax <= vmin:
        vmax = vmin + 1.0

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5.5, 5.0)
    else:
        fig = ax.figure

    if use_contourf:
        levels = np.linspace(vmin, vmax, 350)
        image = ax.contourf(
            q_axis,
            freq_THz,
            np.clip(sed_log, vmin, vmax),
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(q_axis[0], q_axis[-1])
        ax.set_xticks([q_axis[0], q_axis[-1]])
    else:
        image = ax.imshow(
            sed_log,
            cmap=cmap,
            interpolation="hanning",
            aspect="auto",
            origin="lower",
            vmax=vmax,
            vmin=vmin,
        )
        ax.set_xlim([0, len(qpoints) - 1])
        ax.set_xticks([0, len(qpoints) - 1])

    bar = fig.colorbar(image, ax=ax)
    ticks = np.arange(vmin, vmax + 0.01, 2)
    if len(ticks) > 0:
        bar.set_ticks(ticks)
        bar.set_ticklabels([str(int(t)) for t in ticks])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, width=0, length=0, pad=0.6)
    bar.set_label(r"log($S$($\mathbf{q}$, $\omega$))", fontsize=13.5)

    max_freq = float(freq_THz.max() if freq_max is None else freq_max)
    freqs = _freq_ticks(max_freq, freq_interval)
    if use_contourf:
        ax.set_yticks(freqs)
        ax.set_ylim([0, max_freq])
    else:
        ids = np.array([np.argwhere(freq_THz <= f).max() for f in freqs], dtype=int)
        ax.set_yticks(ids)
        ax.set_ylim([0, max_freq / freq_THz.max() * (len(freq_THz) - 1)])
    ax.set_yticklabels(_format_freq_labels(freqs), fontsize=13.5)

    ax.set_xticklabels(q_labels, fontsize=16)
    ax.set_ylabel("Frequency (THz)", fontsize=16)
    ax.set_title(f"SED {component}", fontsize=14)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=650, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _unpack_sed_input(
    result: DipoleSedResult | np.ndarray,
    freq_THz: np.ndarray | None,
    qpoints: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, DipoleSedResult):
        return result.sed, result.freq_THz, result.qpoints
    if freq_THz is None or qpoints is None:
        raise ValueError("freq_THz and qpoints are required when result is an array.")
    return np.asarray(result), freq_THz, qpoints


def _select_component(sed: np.ndarray, component: str) -> np.ndarray:
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of {tuple(_COMPONENTS)}.")

    sed = np.asarray(sed)
    if sed.ndim == 2:
        if component != "total":
            raise ValueError("2D SED data only supports component='total'.")
        return sed
    if sed.ndim != 3 or sed.shape[-1] < 4:
        raise ValueError("SED must have shape (nfreq, nq) or (nfreq, nq, 4).")
    return sed[:, :, _COMPONENTS[component]]


def _q_axis(qpoints: np.ndarray, q_distances: np.ndarray | None) -> np.ndarray:
    if q_distances is not None and len(q_distances) == len(qpoints):
        return np.asarray(q_distances, dtype=np.float64)
    if len(qpoints) == 1:
        return np.array([0.0])

    dq = np.diff(qpoints, axis=0)
    dq -= np.round(dq)
    distances = np.zeros(len(qpoints), dtype=np.float64)
    distances[1:] = np.cumsum(np.linalg.norm(dq, axis=1))
    return distances


def _freq_ticks(max_freq: float, interval: float) -> np.ndarray:
    if interval <= 0:
        raise ValueError("freq_interval must be positive.")
    if max_freq < 0:
        raise ValueError("freq_max must be non-negative.")
    num_ticks = int(np.ceil(max_freq / interval)) + 1
    return np.linspace(0.0, max_freq, num_ticks)


def _format_freq_labels(freqs: np.ndarray) -> list[str]:
    return [f"{f:.1f}".rstrip("0").rstrip(".") for f in freqs]
