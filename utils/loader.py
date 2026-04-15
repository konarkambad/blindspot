"""
loader.py — Data loading and validation utility.

Handles CSV and Excel ingestion with auto-detection of delimiters and encodings.
Returns a clean DataFrame alongside a metadata dictionary for downstream auditing.
"""

import io
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "cp1252"]
CSV_DELIMITERS = [",", "\t", ";", "|"]

MAX_FILE_SIZE_MB  = 200
MAX_COLUMNS       = 500
WARN_COLUMNS      = 100
WARN_ROWS         = 1_000_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_delimiter(raw_bytes: bytes) -> str:
    """
    Sniff the most likely delimiter from the first 4 KB of raw CSV bytes.
    Falls back to comma if nothing conclusive is found.

    Mixed-delimiter files: uses the first non-header data line for scoring
    so that header rows with unusual formatting don't mislead detection.
    """
    sample = raw_bytes[:4096].decode("utf-8", errors="replace")
    lines = [l for l in sample.splitlines() if l.strip()]

    # prefer data row
    probe_line = lines[1] if len(lines) > 1 else (lines[0] if lines else "")

    counts = {delim: probe_line.count(delim) for delim in CSV_DELIMITERS}
    best = max(counts, key=counts.get)

    # require at least one
    return best if counts[best] > 0 else ","


def _read_csv_with_fallback(raw_bytes: bytes) -> pd.DataFrame:
    """
    Try reading a CSV by:
      1. Auto-detecting the delimiter.
      2. Trying each supported encoding in order until one succeeds.

    Raises ValueError if all attempts fail.
    """
    delimiter = _detect_delimiter(raw_bytes)
    last_error = None

    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="warn",
            )
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(f"Could not parse CSV with any supported encoding. Last error: {last_error}")


def _read_excel(raw_bytes: bytes) -> pd.DataFrame:
    """
    Read the first sheet of an Excel file (.xlsx / .xls).
    Raises ValueError if parsing fails.
    """
    try:
        return pd.read_excel(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse Excel file: {exc}")


def _guess_has_header(df: pd.DataFrame) -> bool:
    """
    Heuristic: if pandas assigned generic integer column names (0, 1, 2 …)
    the file probably has no header row. Named columns suggest a header exists.
    """
    return not all(isinstance(col, int) for col in df.columns)


def _build_dtype_summary(df: pd.DataFrame) -> dict:
    """
    Count columns by broad dtype category.
    """
    numeric_count = 0
    categorical_count = 0
    datetime_count = 0
    boolean_count = 0

    for dtype in df.dtypes:
        if pd.api.types.is_bool_dtype(dtype):
            boolean_count += 1
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_count += 1
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_count += 1
        else:
            # non-numeric → categorical
            categorical_count += 1

    return {
        "numeric": numeric_count,
        "categorical": categorical_count,
        "datetime": datetime_count,
        "boolean": boolean_count,
    }


def _all_null_columns(df: pd.DataFrame) -> list:
    """Return names of columns where every value is null."""
    return [col for col in df.columns if df[col].isnull().all()]


def _build_metadata(df: pd.DataFrame) -> dict:
    """
    Assemble the full metadata dictionary from a loaded DataFrame.
    Includes warnings list for non-fatal issues detected during loading.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    warnings = []

    if len(df) == 0:
        warnings.append("File contains a header row but no data rows.")
    if len(df.columns) > WARN_COLUMNS:
        warnings.append(
            f"Dataset has {len(df.columns)} columns — performance may be slow. "
            "Consider narrowing to relevant columns before auditing."
        )
    if len(df) > WARN_ROWS:
        warnings.append(
            f"Dataset has {len(df):,} rows — analysis will be slower. "
            "Results remain valid but some modules sample for performance."
        )
    all_null = _all_null_columns(df)
    if all_null:
        warnings.append(
            f"{len(all_null)} column(s) are entirely null and carry no information: "
            + ", ".join(all_null[:5])
            + ("…" if len(all_null) > 5 else "")
        )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes_summary": _build_dtype_summary(df),
        "memory_usage_mb": round(memory_bytes / (1024 ** 2), 4),
        "has_header": _guess_has_header(df),
        "duplicate_row_count": int(df.duplicated().sum()),
        "all_null_columns": all_null,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(uploaded_file) -> tuple:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : file-like object
        Any object that exposes a `.name` attribute and `.read()` method
        (e.g. a Streamlit UploadedFile, or a plain open() file handle).

    Returns
    -------
    (DataFrame, dict)
        On success: a tuple of (dataframe, metadata_dict).
        metadata_dict keys:
            - row_count          : int
            - column_count       : int
            - column_names       : list[str]
            - dtypes_summary     : dict with counts per broad type
            - memory_usage_mb    : float
            - has_header         : bool (heuristic guess)
            - duplicate_row_count: int

    (None, str)
        On failure: a tuple of (None, human-readable error message).
    """
    try:
        filename: str = getattr(uploaded_file, "name", "unknown")
        raw_bytes: bytes = uploaded_file.read()

        if not raw_bytes:
            return None, "The uploaded file is empty."

        size_mb = len(raw_bytes) / (1024 ** 2)
        if size_mb > MAX_FILE_SIZE_MB:
            return None, (
                f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit. "
                "Please reduce the file size before uploading."
            )

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext in ("xlsx", "xls"):
            df = _read_excel(raw_bytes)
        elif ext == "csv":
            df = _read_csv_with_fallback(raw_bytes)
        else:
            # try CSV, then Excel
            try:
                df = _read_csv_with_fallback(raw_bytes)
            except ValueError:
                df = _read_excel(raw_bytes)

        if df.shape[1] > MAX_COLUMNS:
            return None, (
                f"Dataset has {df.shape[1]} columns, which exceeds the {MAX_COLUMNS}-column limit. "
                "Audit accuracy degrades on very wide datasets. Please select relevant columns first."
            )

        return df, _build_metadata(df)

    except Exception as exc:
        return None, f"Failed to load file: {exc}"
