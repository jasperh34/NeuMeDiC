# load_data.py
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from word2number import w2n  # type: ignore  # pip install word2number

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
GLUCOSE_DIR = os.path.join(DATA_DIR, 'Glucose')
STIM_PATH = os.path.join(DATA_DIR, 'Stimulation.xlsx')


def _normalize_pid(value) -> Optional[str]:
    """
    Return a canonical numeric participant id as a string from many forms.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()

    # If there are digits, use them directly.
    m = re.search(r'(\d+)', s)
    if m:
        return str(int(m.group(1)))  

    # Otherwise try to parse an English number word.
    # Remove decorations like "participant", punctuation, underscores, hyphens, etc.
    letters = re.sub(r'[^A-Za-z\s-]', ' ', s).lower()
    letters = letters.replace('_', ' ').replace('-', ' ')
    letters = re.sub(r'\bparticipant\b', ' ', letters)
    letters = re.sub(r'\s+', ' ', letters).strip()
    if not letters:
        return None
    try:
        return str(int(w2n.word_to_num(letters)))
    except Exception:
        return None


def load_all_csvs(verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load ALL CSVs directly under Data/ into a dict keyed by filename (without .csv).
    """
    dfs: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith('.csv'):
            continue
        key = fn[:-4]
        path = os.path.join(DATA_DIR, fn)
        try:
            df = pd.read_csv(path)
            # Canonical PID where possible
            if 'PID' not in df.columns:
                if 'Id' in df.columns:
                    df['PID'] = df['Id'].apply(_normalize_pid)
                elif 'Participant' in df.columns:
                    df['PID'] = df['Participant'].apply(_normalize_pid)
            dfs[key] = df
            if verbose:
                print(f'Loaded {fn:35s}  {df.shape[0]:>8,d} rows')
        except Exception as exc:
            print(f'{fn}  {exc}')
    return dfs


def load_glucose_csvs(verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Merge all Dexcom Clarity CSVs under Data/Glucose into one DataFrame.
    We keep only time-series rows (those with timestamp & glucose value present).
    Returns columns: [Timestamp (YYYY-MM-DDThh:mm:ss), Glucose Value (mmol/L), PID]
    """
    if not os.path.isdir(GLUCOSE_DIR):
        if verbose:
            print(f'Glucose folder not found: {GLUCOSE_DIR}')
        return None

    ts_col = 'Timestamp (YYYY-MM-DDThh:mm:ss)'
    gv_col = 'Glucose Value (mmol/L)'
    frames: List[pd.DataFrame] = []

    for fn in os.listdir(GLUCOSE_DIR):
        lower = fn.lower()
        if not (lower.startswith('clarity_export') and lower.endswith('.csv')):
            continue

        # Extract a word after "Clarity_Export_" -> e.g., One, Two, Three
        m = re.search(r'clarity[\s_-]?export[\s_-]?([A-Za-z-]+)', fn, flags=re.IGNORECASE)
        if not m:
            if verbose:
                print(f"Could not parse participant word from filename: {fn}")
            continue

        word = re.split(r'[^A-Za-z]+', m.group(1))[0]  # stop at next separator
        pid = _normalize_pid(word)
        if pid is None:
            if verbose:
                print(f"Could not convert '{word}' to number in {fn}")
            continue

        path = os.path.join(GLUCOSE_DIR, fn)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            if verbose:
                print(f'{fn}  {exc}')
            continue

        if ts_col not in df.columns or gv_col not in df.columns:
            if verbose:
                print(f"{fn}: expected columns missing; found {list(df.columns)[:6]}...")
            continue

        sub = df[[ts_col, gv_col]].copy()
        sub[ts_col] = pd.to_datetime(sub[ts_col], errors='coerce')
        sub.dropna(subset=[ts_col, gv_col], inplace=True)
        sub['PID'] = str(pid)
        frames.append(sub)

        if verbose:
            print(f'Loaded {fn:35s}  {sub.shape[0]:>8,d} rows  (PID={pid})')

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(ts_col, inplace=True)
    return out


def load_stimulation(verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load stimulation intervals from Data/Stimulation.xlsx.
    Returns columns: ['PID', 'Start Time', 'End Time'] with datetimes parsed.
    """
    if not os.path.exists(STIM_PATH):
        if verbose:
            print(f"Stimulation file not found: {STIM_PATH}")
        return None

    try:
        df = pd.read_excel(STIM_PATH)
    except Exception as exc:
        if verbose:
            print(f"Error reading Stimulation.xlsx: {exc}")
        return None

    # Normalize column names for flexibility
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id') or cols.get('participant') or list(df.columns)[0]
    start_col = cols.get('start time') or cols.get('start') or list(df.columns)[1]
    end_col = cols.get('end time') or cols.get('end') or list(df.columns)[2]

    out = df[[id_col, start_col, end_col]].copy()
    out.rename(columns={id_col: 'Id', start_col: 'Start Time', end_col: 'End Time'}, inplace=True)

    out['PID'] = out['Id'].apply(_normalize_pid)
    out['Start Time'] = pd.to_datetime(out['Start Time'], errors='coerce')
    out['End Time'] = pd.to_datetime(out['End Time'], errors='coerce')
    out.dropna(subset=['PID', 'Start Time', 'End Time'], inplace=True)

    if verbose:
        print(f"Loaded Stimulation.xlsx with {len(out)} rows")

    return out[['PID', 'Start Time', 'End Time']]


def preview_cols(dfs, key):  
    return list(dfs[key].columns) if key in dfs else []
