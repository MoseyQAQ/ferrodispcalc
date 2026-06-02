from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


__all__ = [
    "DipoleModeResult",
    "DipoleSedResult",
    "calculate_sed",
    "extract_eigen_vector",
    "generate_commensurate_qpath",
    "load_eigen_vector",
    "load_sed",
    "save_eigen_vector",
    "save_sed",
]


@dataclass(frozen=True)
class DipoleSedResult:
    """Result of a local-mode SED calculation.

    Attributes
    ----------
    freq_THz : np.ndarray
        Positive-frequency axis in THz.
    qpoints : np.ndarray
        Reduced q-points used in the spatial Fourier transform.
    sed : np.ndarray
        SED intensity with shape ``(nfreq, nq, 4)``. The last axis stores
        ``x``, ``y``, ``z``, and ``total``.
    primitive_shape : tuple[int, int, int]
        Primitive block size used to group the input grid.
    cell_shape : tuple[int, int, int]
        Number of primitive cells along each grid direction.
    basis_summation : str
        Basis summation rule. ``"incoherent"`` sums intensities over basis
        degrees of freedom; ``"coherent"`` sums amplitudes before squaring.
    """

    freq_THz: np.ndarray
    qpoints: np.ndarray
    sed: np.ndarray
    primitive_shape: tuple[int, int, int]
    cell_shape: tuple[int, int, int]
    basis_summation: str = "incoherent"


@dataclass(frozen=True)
class DipoleModeResult:
    """Result of a basis-resolved local-mode extraction.

    Attributes
    ----------
    mode_movie : np.ndarray
        Real-space mode movie with shape ``(nphase, nx, ny, nz, 3)``.
    phases : np.ndarray
        Phase samples in radians with shape ``(nphase,)``.
    evec_basis : np.ndarray
        Complex basis-resolved eigenvector with shape ``(px, py, pz, 3)``.
    amplitude_basis : np.ndarray
        Complex, unnormalized Fourier amplitude after gauge fixing, with shape
        ``(px, py, pz, 3)``.
    qpoint : np.ndarray
        Reduced-coordinate q-point used for extraction.
    requested_freq_THz : float
        Target frequency requested by the user.
    actual_freq_THz : float
        Frequency actually used. This can differ from the requested value for
        ``freq_method="nearest_fft"``.
    primitive_shape : tuple[int, int, int]
        Primitive block size used to group the input grid.
    cell_shape : tuple[int, int, int]
        Number of primitive cells along each grid direction.
    freq_method : str
        Frequency extraction method, either ``"nearest_fft"`` or ``"direct"``.
    gauge : str
        Gauge-fixing method, either ``"max_real"`` or ``"none"``.
    normalize : bool
        Whether ``evec_basis`` was normalized by ``sqrt(sum(abs(A)**2))``.
    remove_mean : bool
        Whether per-grid-point time means were removed before extraction.
    """

    mode_movie: np.ndarray
    phases: np.ndarray
    evec_basis: np.ndarray
    amplitude_basis: np.ndarray
    qpoint: np.ndarray
    requested_freq_THz: float
    actual_freq_THz: float
    primitive_shape: tuple[int, int, int]
    cell_shape: tuple[int, int, int]
    freq_method: str
    gauge: str
    normalize: bool
    remove_mean: bool


def calculate_sed(
    field: np.ndarray,
    dt_ps: float,
    qpoints: np.ndarray,
    primitive_shape: tuple[int, int, int] = (1, 1, 1),
    num_splits: int = 1,
    remove_mean: bool = False,
    n_jobs: int = 1,
    parallel_backend: str = "threading",
    *,
    basis_summation: str = "incoherent",
) -> DipoleSedResult:
    """Calculate a local-mode fluctuation spectrum for a gridded trajectory.

    The input field is assumed to be fixed on a regular grid of local modes,
    such as Ti-centered displacement, local dipole, local polarization, or the
    corresponding time derivative. Users must provide the reduced q-points
    explicitly; use :func:`generate_commensurate_qpath` when a path should be
    sampled by q-points commensurate with the simulation cell.

    Parameters
    ----------
    field : np.ndarray
        Input trajectory with shape ``(nframe, nx, ny, nz, 3)``. The last axis
        stores the Cartesian ``x``, ``y``, and ``z`` components. If ``dt_ps`` is
        in ps and the field is from LAMMPS metal units, displacement-like input
        is typically in Angstrom and velocity-like input is in Angstrom/ps.
    dt_ps : float
        Time interval between stored frames in ps. The returned frequency axis
        is in THz because ``1 / ps = 1 THz``.
    qpoints : np.ndarray
        Reduced q-points with shape ``(nq, 3)``. The spatial phase is
        ``exp(+i * 2*pi * dot(q, cell_index + basis_offset))``, where
        ``basis_offset`` is the fractional position inside the primitive block.
        The opposite sign gives the same intensity after taking ``|A|^2``.
    primitive_shape : tuple[int, int, int], optional
        Primitive block size in grid units. For example, ``(1, 1, 5)`` groups
        five local modes into each primitive cell along z. Defaults to
        ``(1, 1, 1)``.
    num_splits : int, optional
        Number of equal time blocks used for block averaging. ``nframe`` must
        be divisible by this value. Defaults to ``1``.
    remove_mean : bool, optional
        If ``True``, subtract the per-block time average at every grid point
        and component before the FFT. This is usually appropriate for a
        displacement-like field to remove the static/DC component. Velocity-like
        fields are usually used as-is. Defaults to ``False``.
    n_jobs : int, optional
        Number of q-points to evaluate in parallel with joblib. ``1`` means
        serial execution; negative values follow joblib's convention. Defaults
        to ``1``.
    parallel_backend : str, optional
        Joblib backend. The default ``"threading"`` avoids copying the large
        block array into worker processes.
    basis_summation : {"incoherent", "coherent"}, optional
        Basis summation rule. ``"incoherent"`` computes
        ``sum_b |U_b(q, omega)|^2`` and is the default local-fluctuation
        spectrum. ``"coherent"`` computes ``|sum_b U_b(q, omega)|^2`` and
        preserves basis interference, closer to a structure-factor-like
        representation.

    Returns
    -------
    DipoleSedResult
        Frequency axis, q-points, SED intensity, and grid metadata.

    Notes
    -----
    The normalization is ``1 / (Nt * Ncell)`` before averaging over time
    blocks, where ``Nt`` is the number of frames per block and ``Ncell`` is the
    number of primitive cells. No mass weighting is applied, so this is not a
    strict mass-weighted phonon SED unless the input field and normalization are
    defined accordingly. The spatial Fourier transform uses the full local-mode
    position within the current folded-cell representation and, by default,
    sums basis intensities incoherently.
    """

    basis_summation = _validate_basis_summation(basis_summation)

    field, qpoints, primitive_shape, nframe, cell_shape = _check_parameter(
        field=field,
        dt_ps=dt_ps,
        qpoints=qpoints,
        primitive_shape=primitive_shape,
        num_splits=num_splits,
        n_jobs=n_jobs,
    )

    # 1. Build primitive-cell coordinates and frequency buffers.
    cell_indices = _make_cell_indices(cell_shape)
    basis_offsets = _make_basis_offsets(primitive_shape)
    nt = nframe // num_splits
    ncell = cell_indices.shape[0]
    norm = 1.0 / (float(nt) * float(ncell))

    freq_full = np.fft.fftfreq(nt, d=dt_ps)
    freq_pos, _ = _fold_positive_frequencies(
        np.zeros((nt, len(qpoints), 4), dtype=np.float64),
        freq_full,
    )
    sed_accum = np.zeros((len(freq_pos), len(qpoints), 4), dtype=np.float64)

    # 2. Accumulate block-averaged SED. Each progress step is one q-point block.
    progress = tqdm(
        total=num_splits * len(qpoints),
        desc="Computing SED",
        unit="qpoint",
        dynamic_ncols=True,
    )
    try:
        for split_idx in range(num_splits):
            start = split_idx * nt
            stop = start + nt
            block = field[start:stop]
            block_basis = _reshape_grid_to_basis(block, primitive_shape)

            if remove_mean:
                block_basis = block_basis - block_basis.mean(axis=0, keepdims=True)
            else:
                block_basis = np.asarray(block_basis)

            sed_full = _compute_block_sed(
                block_basis,
                qpoints,
                cell_indices,
                basis_offsets,
                norm,
                basis_summation=basis_summation,
                progress=progress,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )
            _, sed_pos = _fold_positive_frequencies(sed_full, freq_full)
            sed_accum += sed_pos
    finally:
        progress.close()

    # 3. Average over time blocks and return the positive-frequency spectrum.
    sed_avg = sed_accum / float(num_splits)
    return DipoleSedResult(
        freq_THz=freq_pos,
        qpoints=qpoints,
        sed=sed_avg,
        primitive_shape=primitive_shape,
        cell_shape=cell_shape,
        basis_summation=basis_summation,
    )


def extract_eigen_vector(
    field: np.ndarray,
    dt_ps: float,
    qpoint: np.ndarray,
    freq_THz: float,
    primitive_shape: tuple[int, int, int] = (1, 1, 1),
    nphase: int = 24,
    remove_mean: bool = True,
    freq_method: str = "nearest_fft",
    gauge: str = "max_real",
    normalize: bool = True,
) -> DipoleModeResult:
    """Extract a basis-resolved complex mode and rebuild a real-space movie.

    The extraction uses the full local-mode position,
    ``cell_index + basis_offset``, in the spatial Fourier transform. It keeps
    the complex amplitude before any basis summation or absolute-square
    operation, then reconstructs the selected ``(q, frequency)`` mode over one
    artificial vibration period.

    Parameters
    ----------
    field : np.ndarray
        Input trajectory with shape ``(nframe, nx, ny, nz, 3)``.
    dt_ps : float
        Time interval between stored frames in ps.
    qpoint : np.ndarray
        Single reduced-coordinate q-point with shape ``(3,)``.
    freq_THz : float
        Target frequency in THz.
    primitive_shape : tuple[int, int, int], optional
        Primitive block size in grid units. This should match the SED
        calculation used to identify the selected peak.
    nphase : int, optional
        Number of phase frames in the returned one-period movie.
    remove_mean : bool, optional
        If ``True``, remove the per-grid-point time average before extracting
        the mode. Defaults to ``True``.
    freq_method : {"nearest_fft", "direct"}, optional
        ``"nearest_fft"`` uses the closest non-negative FFT frequency bin.
        ``"direct"`` projects the trajectory directly at ``freq_THz``.
    gauge : {"max_real", "none"}, optional
        ``"max_real"`` rotates the global complex phase so the largest
        eigenvector component is positive real. ``"none"`` leaves the phase as
        extracted.
    normalize : bool, optional
        If ``True``, normalize the eigenvector by
        ``sqrt(sum(abs(A)**2))``. Defaults to ``True``.

    Returns
    -------
    DipoleModeResult
        Mode movie, basis eigenvector, raw basis amplitude, and metadata.

    Notes
    -----
    The returned ``mode_movie`` is a normalized local-mode pattern when
    ``normalize=True``. Its amplitude is therefore not the physical MD
    vibration amplitude; use ``amplitude_basis`` or the SED intensity to judge
    mode strength.
    """

    (
        field,
        qpoint,
        primitive_shape,
        _nframe,
        cell_shape,
        nphase,
        freq_THz,
        freq_method,
        gauge,
    ) = _check_mode_parameter(
        field=field,
        dt_ps=dt_ps,
        qpoint=qpoint,
        freq_THz=freq_THz,
        primitive_shape=primitive_shape,
        nphase=nphase,
        freq_method=freq_method,
        gauge=gauge,
    )

    remove_mean = _validate_bool(remove_mean, name="remove_mean")
    normalize = _validate_bool(normalize, name="normalize")
    cell_indices = _make_cell_indices(cell_shape)
    basis_offsets = _make_basis_offsets(primitive_shape)
    block_basis = _reshape_grid_to_basis(field, primitive_shape)

    if remove_mean:
        block_basis = block_basis - block_basis.mean(axis=0, keepdims=True)
    else:
        block_basis = np.asarray(block_basis)

    q_signal = _compute_q_signal(
        block_basis,
        cell_indices,
        basis_offsets,
        qpoint,
    )
    amplitude, actual_freq_THz = _extract_frequency_amplitude(
        q_signal=q_signal,
        dt_ps=dt_ps,
        freq_THz=freq_THz,
        freq_method=freq_method,
    )
    evec = _normalize_eigenvector(amplitude, normalize=normalize)
    evec, amplitude = _apply_mode_gauge(evec, amplitude, gauge=gauge)

    px, py, pz = primitive_shape
    evec_basis = evec.reshape(px, py, pz, 3)
    amplitude_basis = amplitude.reshape(px, py, pz, 3)
    phases = np.linspace(0.0, 2.0 * np.pi, nphase, endpoint=False)
    mode_basis = _reconstruct_mode_basis(
        evec=evec,
        phases=phases,
        qpoint=qpoint,
        cell_indices=cell_indices,
        basis_offsets=basis_offsets,
    )
    mode_movie = _reshape_basis_to_grid(
        mode_basis,
        cell_shape=cell_shape,
        primitive_shape=primitive_shape,
    )

    return DipoleModeResult(
        mode_movie=mode_movie,
        phases=phases,
        evec_basis=evec_basis,
        amplitude_basis=amplitude_basis,
        qpoint=qpoint,
        requested_freq_THz=freq_THz,
        actual_freq_THz=actual_freq_THz,
        primitive_shape=primitive_shape,
        cell_shape=cell_shape,
        freq_method=freq_method,
        gauge=gauge,
        normalize=normalize,
        remove_mean=remove_mean,
    )


def generate_commensurate_qpath(
    q_path: np.ndarray,
    cell_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate q-points along a path that is commensurate with a cell grid.

    For cell shape ``(Nx, Ny, Nz)``, allowed reduced q-points satisfy
    ``q * (Nx, Ny, Nz)`` being integer-valued.

    Returns
    -------
    qpoints, q_distances
        Reduced q-points and cumulative reduced-coordinate distances.
    """

    q_path = np.asarray(q_path, dtype=np.float64)
    if q_path.ndim != 2 or q_path.shape[1] != 3 or q_path.shape[0] < 2:
        raise ValueError("q_path must have shape (n_endpoint >= 2, 3).")

    cell_shape = tuple(int(v) for v in cell_shape)
    if any(v <= 0 for v in cell_shape):
        raise ValueError("cell_shape values must be positive.")

    qpoints = []
    for i in range(q_path.shape[0] - 1):
        start = q_path[i]
        end = q_path[i + 1]
        fractions = _allowed_path_fractions(start, end, cell_shape)
        segment = [start + float(f) * (end - start) for f in fractions]
        if qpoints and segment and np.allclose(qpoints[-1], segment[0]):
            segment = segment[1:]
        qpoints.extend(segment)

    if not qpoints:
        raise ValueError("No commensurate q-points were generated.")

    qpoints_arr = np.asarray(qpoints, dtype=np.float64)
    qpoints_arr[np.isclose(qpoints_arr, 0.0, atol=1e-15)] = 0.0
    q_distances = _reduced_q_distances(qpoints_arr)
    return qpoints_arr, q_distances


def save_sed(result: DipoleSedResult, filename: str | Path) -> None:
    """Save SED arrays and grid metadata to a compressed npz file."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    basis_summation = _validate_basis_summation(result.basis_summation)
    np.savez_compressed(
        filename,
        freq_THz=result.freq_THz,
        qpoints=result.qpoints,
        sed=result.sed,
        primitive_shape=np.asarray(result.primitive_shape, dtype=np.int64),
        cell_shape=np.asarray(result.cell_shape, dtype=np.int64),
        basis_summation=np.asarray(basis_summation),
    )


def load_sed(filename: str | Path) -> DipoleSedResult:
    """Load a :class:`DipoleSedResult` saved by :func:`save_sed`."""

    data = np.load(filename)
    required = {"freq_THz", "qpoints", "sed", "primitive_shape", "cell_shape"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing SED fields in {filename}: {sorted(missing)}")

    return DipoleSedResult(
        freq_THz=data["freq_THz"],
        qpoints=data["qpoints"],
        sed=data["sed"],
        primitive_shape=tuple(int(v) for v in data["primitive_shape"]),
        cell_shape=tuple(int(v) for v in data["cell_shape"]),
        basis_summation=_validate_basis_summation(
            _load_npz_string(data, "basis_summation", default="incoherent")
        ),
    )


def save_eigen_vector(result: DipoleModeResult, filename: str | Path) -> None:
    """Save an extracted mode result to a compressed npz file."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    freq_method = _validate_freq_method(result.freq_method)
    gauge = _validate_gauge(result.gauge)
    np.savez_compressed(
        filename,
        mode_movie=result.mode_movie,
        phases=result.phases,
        evec_basis=result.evec_basis,
        amplitude_basis=result.amplitude_basis,
        qpoint=result.qpoint,
        requested_freq_THz=np.asarray(result.requested_freq_THz, dtype=np.float64),
        actual_freq_THz=np.asarray(result.actual_freq_THz, dtype=np.float64),
        primitive_shape=np.asarray(result.primitive_shape, dtype=np.int64),
        cell_shape=np.asarray(result.cell_shape, dtype=np.int64),
        freq_method=np.asarray(freq_method),
        gauge=np.asarray(gauge),
        normalize=np.asarray(bool(result.normalize)),
        remove_mean=np.asarray(bool(result.remove_mean)),
    )


def load_eigen_vector(filename: str | Path) -> DipoleModeResult:
    """Load a :class:`DipoleModeResult` saved by :func:`save_eigen_vector`."""

    data = np.load(filename)
    required = {
        "mode_movie",
        "phases",
        "evec_basis",
        "amplitude_basis",
        "qpoint",
        "requested_freq_THz",
        "actual_freq_THz",
        "primitive_shape",
        "cell_shape",
        "freq_method",
        "gauge",
        "normalize",
        "remove_mean",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing mode fields in {filename}: {sorted(missing)}")

    return DipoleModeResult(
        mode_movie=data["mode_movie"],
        phases=data["phases"],
        evec_basis=data["evec_basis"],
        amplitude_basis=data["amplitude_basis"],
        qpoint=data["qpoint"],
        requested_freq_THz=float(np.asarray(data["requested_freq_THz"]).item()),
        actual_freq_THz=float(np.asarray(data["actual_freq_THz"]).item()),
        primitive_shape=tuple(int(v) for v in data["primitive_shape"]),
        cell_shape=tuple(int(v) for v in data["cell_shape"]),
        freq_method=_validate_freq_method(
            _load_npz_string(data, "freq_method", default="nearest_fft")
        ),
        gauge=_validate_gauge(_load_npz_string(data, "gauge", default="max_real")),
        normalize=_load_npz_bool(data, "normalize"),
        remove_mean=_load_npz_bool(data, "remove_mean"),
    )


def _compute_block_sed(
    block_basis: np.ndarray,
    qpoints: np.ndarray,
    cell_indices: np.ndarray,
    basis_offsets: np.ndarray,
    normalization_factor: float,
    basis_summation: str,
    progress=None,
    n_jobs: int = 1,
    parallel_backend: str = "threading",
) -> np.ndarray:
    """Compute one-block full-frequency SED."""

    if block_basis.ndim != 4 or block_basis.shape[-1] != 3:
        raise ValueError("block_basis must have shape (nt, ncell, nbasis, 3).")

    nt, ncell, nbasis, _ = block_basis.shape
    if cell_indices.shape != (ncell, 3):
        raise ValueError(
            "cell_indices shape must be (ncell, 3), matching block_basis."
        )
    if basis_offsets.shape != (nbasis, 3):
        raise ValueError(
            "basis_offsets shape must be (nbasis, 3), matching block_basis."
        )

    sed_comp = np.empty((nt, len(qpoints), 3), dtype=np.float64)

    if n_jobs == 1:
        for iq, q in enumerate(qpoints):
            sed_comp[:, iq, :] = _compute_q_sed(
                block_basis,
                cell_indices,
                basis_offsets,
                q,
                basis_summation,
            )
            if progress is not None:
                progress.update(1)
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend=parallel_backend,
            return_as="generator_unordered",
        )(
            delayed(_compute_q_sed_indexed)(
                iq,
                block_basis,
                cell_indices,
                basis_offsets,
                q,
                basis_summation,
            )
            for iq, q in enumerate(qpoints)
        )
        for iq, sed_q in results:
            sed_comp[:, iq, :] = sed_q
            if progress is not None:
                progress.update(1)

    sed_comp *= normalization_factor
    sed_total = sed_comp.sum(axis=2, keepdims=True)
    return np.concatenate((sed_comp, sed_total), axis=2)


def _reshape_grid_to_basis(
    field: np.ndarray,
    primitive_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reshape ``(nt, nx, ny, nz, 3)`` to ``(nt, ncell, nbasis, 3)``."""

    _validate_field(field)
    px, py, pz = _validate_primitive_shape(primitive_shape)
    nt, nx, ny, nz, _ = field.shape
    if nx % px or ny % py or nz % pz:
        raise ValueError(
            f"primitive_shape={primitive_shape} must divide grid shape "
            f"({nx}, {ny}, {nz})."
        )

    nx_cell, ny_cell, nz_cell = nx // px, ny // py, nz // pz
    reshaped = field.reshape(
        nt,
        nx_cell,
        px,
        ny_cell,
        py,
        nz_cell,
        pz,
        3,
    )
    reordered = reshaped.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    return reordered.reshape(
        nt,
        nx_cell * ny_cell * nz_cell,
        px * py * pz,
        3,
    )


def _reshape_basis_to_grid(
    basis_field: np.ndarray,
    cell_shape: tuple[int, int, int],
    primitive_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reshape ``(nt, ncell, nbasis, 3)`` back to grid form."""

    basis_field = np.asarray(basis_field)
    if basis_field.ndim != 4 or basis_field.shape[-1] != 3:
        raise ValueError("basis_field must have shape (nt, ncell, nbasis, 3).")

    nx_cell, ny_cell, nz_cell = _validate_cell_shape(cell_shape)
    px, py, pz = _validate_primitive_shape(primitive_shape)
    nframe_like, ncell, nbasis, _ = basis_field.shape
    if ncell != nx_cell * ny_cell * nz_cell:
        raise ValueError("basis_field ncell does not match cell_shape.")
    if nbasis != px * py * pz:
        raise ValueError("basis_field nbasis does not match primitive_shape.")

    reshaped = basis_field.reshape(
        nframe_like,
        nx_cell,
        ny_cell,
        nz_cell,
        px,
        py,
        pz,
        3,
    )
    reordered = reshaped.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    return reordered.reshape(
        nframe_like,
        nx_cell * px,
        ny_cell * py,
        nz_cell * pz,
        3,
    )


def _make_cell_indices(cell_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = _validate_cell_shape(cell_shape)

    grid = np.indices((nx, ny, nz), dtype=np.float64)
    return np.moveaxis(grid, 0, -1).reshape(-1, 3)


def _make_basis_offsets(primitive_shape: tuple[int, int, int]) -> np.ndarray:
    px, py, pz = _validate_primitive_shape(primitive_shape)
    grid = np.indices((px, py, pz), dtype=np.float64)
    offsets = np.moveaxis(grid, 0, -1).reshape(-1, 3)
    return offsets / np.asarray((px, py, pz), dtype=np.float64)


def _reduced_q_distances(qpoints: np.ndarray) -> np.ndarray:
    qpoints = _validate_qpoints(qpoints)
    distances = np.zeros(qpoints.shape[0], dtype=np.float64)
    for i in range(1, qpoints.shape[0]):
        dq = qpoints[i] - qpoints[i - 1]
        dq -= np.round(dq)
        distances[i] = distances[i - 1] + float(np.linalg.norm(dq))
    return distances


def _fold_positive_frequencies(
    sed_full: np.ndarray,
    freq_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    freq_full = np.asarray(freq_full)
    if sed_full.shape[0] != freq_full.shape[0]:
        raise ValueError("sed_full and freq_full must have the same first axis.")

    n = freq_full.shape[0]
    if n < 2:
        return freq_full.copy(), sed_full.copy()

    if n % 2 == 0:
        n_pos = n // 2
        sed_pos = sed_full[:n_pos].copy()
        neg = sed_full[:n_pos:-1]
    else:
        n_pos = n // 2 + 1
        sed_pos = sed_full[:n_pos].copy()
        neg = sed_full[: n // 2 : -1]

    if n_pos > 1:
        sed_pos[1:] = 0.5 * (sed_pos[1:] + neg)
    return freq_full[:n_pos].copy(), sed_pos


def _check_parameter(
    field: np.ndarray,
    dt_ps: float,
    qpoints: np.ndarray,
    primitive_shape: tuple[int, int, int],
    num_splits: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], int, tuple[int, int, int]]:
    field = np.asarray(field)
    qpoints = _validate_qpoints(qpoints)
    primitive_shape = _validate_primitive_shape(primitive_shape)
    _validate_field(field)

    if dt_ps <= 0.0:
        raise ValueError("dt_ps must be positive.")
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1.")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0.")

    nframe, nx, ny, nz, _ = field.shape
    if nframe % num_splits != 0:
        raise ValueError(
            f"nframe={nframe} is not divisible by num_splits={num_splits}. "
            "Slice or trim the trajectory before calling calculate_sed."
        )

    px, py, pz = primitive_shape
    if nx % px or ny % py or nz % pz:
        raise ValueError(
            f"primitive_shape={primitive_shape} must divide grid shape "
            f"({nx}, {ny}, {nz})."
        )

    cell_shape = (nx // px, ny // py, nz // pz)
    return field, qpoints, primitive_shape, nframe, cell_shape


def _check_mode_parameter(
    field: np.ndarray,
    dt_ps: float,
    qpoint: np.ndarray,
    freq_THz: float,
    primitive_shape: tuple[int, int, int],
    nphase: int,
    freq_method: str,
    gauge: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int],
    int,
    tuple[int, int, int],
    int,
    float,
    str,
    str,
]:
    field = np.asarray(field)
    qpoint = np.asarray(qpoint, dtype=np.float64)
    primitive_shape = _validate_primitive_shape(primitive_shape)
    freq_method = _validate_freq_method(freq_method)
    gauge = _validate_gauge(gauge)
    _validate_field(field)

    if qpoint.shape != (3,):
        raise ValueError("qpoint must have shape (3,).")
    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        raise ValueError("dt_ps must be positive.")

    freq_THz = float(freq_THz)
    if not np.isfinite(freq_THz) or freq_THz < 0.0:
        raise ValueError("freq_THz must be non-negative.")

    if isinstance(nphase, bool):
        raise ValueError("nphase must be an integer >= 1.")
    if isinstance(nphase, (float, np.floating)) and not float(nphase).is_integer():
        raise ValueError("nphase must be an integer >= 1.")
    nphase = int(nphase)
    if nphase < 1:
        raise ValueError("nphase must be >= 1.")

    nframe, nx, ny, nz, _ = field.shape
    px, py, pz = primitive_shape
    if nx % px or ny % py or nz % pz:
        raise ValueError(
            f"primitive_shape={primitive_shape} must divide grid shape "
            f"({nx}, {ny}, {nz})."
        )

    cell_shape = (nx // px, ny // py, nz // pz)
    return (
        field,
        qpoint,
        primitive_shape,
        nframe,
        cell_shape,
        nphase,
        freq_THz,
        freq_method,
        gauge,
    )


def _compute_q_sed_indexed(
    iq: int,
    block_basis: np.ndarray,
    cell_indices: np.ndarray,
    basis_offsets: np.ndarray,
    qpoint: np.ndarray,
    basis_summation: str,
) -> tuple[int, np.ndarray]:
    return iq, _compute_q_sed(
        block_basis,
        cell_indices,
        basis_offsets,
        qpoint,
        basis_summation,
    )


def _compute_q_sed(
    block_basis: np.ndarray,
    cell_indices: np.ndarray,
    basis_offsets: np.ndarray,
    qpoint: np.ndarray,
    basis_summation: str,
) -> np.ndarray:
    q_signal = _compute_q_signal(block_basis, cell_indices, basis_offsets, qpoint)
    amp = np.fft.fft(q_signal, axis=0)

    if basis_summation == "incoherent":
        return np.sum(np.abs(amp) ** 2, axis=1).real
    if basis_summation == "coherent":
        return (np.abs(np.sum(amp, axis=1)) ** 2).real
    raise ValueError("basis_summation must be 'incoherent' or 'coherent'.")


def _compute_q_signal(
    block_basis: np.ndarray,
    cell_indices: np.ndarray,
    basis_offsets: np.ndarray,
    qpoint: np.ndarray,
) -> np.ndarray:
    """Spatial Fourier transform at one q-point, keeping the basis axis."""

    phase = np.exp(2.0j * np.pi * (cell_indices @ qpoint))
    q_signal = np.einsum("tlbc,l->tbc", block_basis, phase, optimize=True)
    basis_phase = np.exp(2.0j * np.pi * (basis_offsets @ qpoint))
    return q_signal * basis_phase[None, :, None]


def _extract_frequency_amplitude(
    q_signal: np.ndarray,
    dt_ps: float,
    freq_THz: float,
    freq_method: str,
) -> tuple[np.ndarray, float]:
    if freq_method == "nearest_fft":
        freq_full = np.fft.fftfreq(q_signal.shape[0], d=dt_ps)
        candidate_indices = np.where(freq_full >= 0.0)[0]
        ifreq = candidate_indices[
            np.argmin(np.abs(freq_full[candidate_indices] - freq_THz))
        ]
        amp = np.fft.fft(q_signal, axis=0)
        return amp[ifreq], float(freq_full[ifreq])

    if freq_method == "direct":
        time_ps = np.arange(q_signal.shape[0], dtype=np.float64) * dt_ps
        time_phase = np.exp(-2.0j * np.pi * freq_THz * time_ps)
        amp = np.einsum("tbc,t->bc", q_signal, time_phase, optimize=True)
        return amp, float(freq_THz)

    raise ValueError("freq_method must be 'nearest_fft' or 'direct'.")


def _normalize_eigenvector(amplitude: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return amplitude.copy()

    norm = float(np.sqrt(np.sum(np.abs(amplitude) ** 2)))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("The selected (q, freq) has nearly zero amplitude.")
    return amplitude / norm


def _apply_mode_gauge(
    evec: np.ndarray,
    amplitude: np.ndarray,
    gauge: str,
) -> tuple[np.ndarray, np.ndarray]:
    if gauge == "none":
        return evec, amplitude

    if gauge == "max_real":
        idx = np.unravel_index(np.argmax(np.abs(evec)), evec.shape)
        phase0 = np.angle(evec[idx])
        factor = np.exp(-1.0j * phase0)
        return evec * factor, amplitude * factor

    raise ValueError("gauge must be 'max_real' or 'none'.")


def _reconstruct_mode_basis(
    evec: np.ndarray,
    phases: np.ndarray,
    qpoint: np.ndarray,
    cell_indices: np.ndarray,
    basis_offsets: np.ndarray,
) -> np.ndarray:
    full_positions = cell_indices[:, None, :] + basis_offsets[None, :, :]
    inverse_phase = np.exp(-2.0j * np.pi * (full_positions @ qpoint))
    return np.real(
        np.exp(1.0j * phases[:, None, None, None])
        * inverse_phase[None, :, :, None]
        * evec[None, None, :, :]
    )


def _allowed_path_fractions(
    start: np.ndarray,
    end: np.ndarray,
    cell_shape: tuple[int, int, int],
) -> list[Fraction]:
    if np.allclose(start, end):
        return [Fraction(0, 1)]

    start_f = [Fraction(float(x)).limit_denominator(1_000_000) for x in start]
    end_f = [Fraction(float(x)).limit_denominator(1_000_000) for x in end]

    possible = None
    for s, e, ncell in zip(start_f, end_f, cell_shape):
        a = s * ncell
        b = (e - s) * ncell
        values = _solve_allowed_fractions_1d(a, b)
        if values is None:
            continue
        if not values:
            return []
        values = set(values)
        possible = values if possible is None else possible & values

    if possible is None:
        return [Fraction(0, 1)]
    return sorted(possible)


def _solve_allowed_fractions_1d(a: Fraction, b: Fraction) -> list[Fraction] | None:
    if b == 0:
        return None if a.denominator == 1 else []

    low, high = (a, a + b) if b > 0 else (a + b, a)
    n_min = math.floor(float(low))
    n_max = math.ceil(float(high))

    values = []
    for integer_value in range(n_min, n_max + 1):
        f = Fraction(integer_value - a, b)
        if 0 <= f <= 1:
            values.append(f)
    return values


def _load_npz_string(data, name: str, default: str) -> str:
    if name not in data.files:
        return default

    value = np.asarray(data[name])
    if value.shape == ():
        item = value.item()
    elif value.size == 1:
        item = value.reshape(-1)[0].item()
    else:
        raise ValueError(f"NPZ field {name!r} must contain a single string.")

    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _load_npz_bool(data, name: str) -> bool:
    value = np.asarray(data[name])
    if value.shape == ():
        return bool(value.item())
    if value.size == 1:
        return bool(value.reshape(-1)[0].item())
    raise ValueError(f"NPZ field {name!r} must contain a single boolean.")


def _validate_cell_shape(cell_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(cell_shape) != 3:
        raise ValueError("cell_shape must contain exactly three integers.")
    cell_shape = tuple(int(v) for v in cell_shape)
    if any(v <= 0 for v in cell_shape):
        raise ValueError("cell_shape values must be positive.")
    return cell_shape


def _validate_bool(value: bool, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean.")


def _validate_basis_summation(basis_summation: str) -> str:
    if not isinstance(basis_summation, str):
        raise ValueError("basis_summation must be 'incoherent' or 'coherent'.")

    basis_summation = basis_summation.strip().lower()
    if basis_summation not in {"incoherent", "coherent"}:
        raise ValueError("basis_summation must be 'incoherent' or 'coherent'.")
    return basis_summation


def _validate_freq_method(freq_method: str) -> str:
    if not isinstance(freq_method, str):
        raise ValueError("freq_method must be 'nearest_fft' or 'direct'.")

    freq_method = freq_method.strip().lower()
    if freq_method not in {"nearest_fft", "direct"}:
        raise ValueError("freq_method must be 'nearest_fft' or 'direct'.")
    return freq_method


def _validate_gauge(gauge: str) -> str:
    if not isinstance(gauge, str):
        raise ValueError("gauge must be 'max_real' or 'none'.")

    gauge = gauge.strip().lower()
    if gauge not in {"max_real", "none"}:
        raise ValueError("gauge must be 'max_real' or 'none'.")
    return gauge


def _validate_field(field: np.ndarray) -> None:
    if field.ndim != 5 or field.shape[-1] != 3:
        raise ValueError("field must have shape (nframe, nx, ny, nz, 3).")
    if field.shape[0] < 2:
        raise ValueError("field must contain at least two time frames.")
    if min(field.shape[1:4]) <= 0:
        raise ValueError("grid dimensions must be positive.")


def _validate_qpoints(qpoints: np.ndarray) -> np.ndarray:
    qpoints = np.asarray(qpoints, dtype=np.float64)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("qpoints must have shape (nq, 3).")
    if qpoints.shape[0] < 1:
        raise ValueError("qpoints must contain at least one point.")
    return qpoints


def _validate_primitive_shape(
    primitive_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(primitive_shape) != 3:
        raise ValueError("primitive_shape must contain exactly three integers.")
    primitive_shape = tuple(int(v) for v in primitive_shape)
    if any(v <= 0 for v in primitive_shape):
        raise ValueError("primitive_shape values must be positive.")
    return primitive_shape
