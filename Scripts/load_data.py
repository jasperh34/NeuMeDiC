# load_data.py
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import datetime as dt

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
GLUCOSE_DIR = os.path.join(DATA_DIR, 'Glucose')

# Visits
VISITS_DIR = os.path.join(DATA_DIR, 'Visits')
VISITS_PATH = os.path.join(VISITS_DIR, 'VisitDatesTimes.xlsx')

# Strict P-code like "P01", "p2", "P 003"
PID_PCODE_RE = re.compile(r'^\s*[Pp]\s*0*(\d+)\s*$')


def _pid_from_pcode(value) -> Optional[str]:
    """
    ONLY accept IDs like 'P01', 'p2', 'P 003'. Return numeric string ('1','2','3').
    Anything else (e.g., 'Participant 1', '1', words) => None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    m = PID_PCODE_RE.match(s)
    return str(int(m.group(1))) if m else None


def load_all_csvs(verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load ALL CSVs directly under Data/ into a dict keyed by filename (without .csv).

    Rows are kept ONLY if their participant ID follows the new convention (P##) in a
    column named 'Id' or 'PID'. Legacy labels like 'Participant 1' are dropped.
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

            created_pid = False
            if 'Id' in df.columns:
                df['PID'] = df['Id'].apply(_pid_from_pcode)
                created_pid = True
            elif 'PID' in df.columns:
                df['PID'] = df['PID'].apply(_pid_from_pcode)
                created_pid = True
            # ignore legacy 'Participant' columns intentionally

            if created_pid:
                before = len(df)
                df.dropna(subset=['PID'], inplace=True)
                if verbose and len(df) != before:
                    print(f"{fn}: dropped {before - len(df)} rows without P## IDs")

            dfs[key] = df
            if verbose:
                print(f'Loaded {fn:35s}  {df.shape[0]:>8,d} rows')
        except Exception as exc:
            print(f'{fn}  {exc}')
    return dfs


def _find_glucose_columns(columns: List[str]):
    """Return (timestamp_col, glucose_col) from a Dexcom CSV in a tolerant way."""
    ts_col = None
    gv_col = None
    for c in columns:
        cl = str(c).lower()
        if ts_col is None and cl.startswith('timestamp'):
            ts_col = c
        if gv_col is None and 'glucose value' in cl:
            gv_col = c
    return ts_col, gv_col


def load_glucose_csvs(verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Merge all Dexcom Clarity CSVs under Data/Glucose into one DataFrame.
    Standardizes columns to:
      - 'Timestamp' (datetime64[ns])
      - 'Glucose (mmol/L)' (float)
      - 'PID' (string)
    Extract PID from filename: ..._(\\d{3})_(Sonic|SonicStudyVNS)...
    """
    if not os.path.isdir(GLUCOSE_DIR):
        if verbose:
            print(f'[glucose] Glucose folder not found: {GLUCOSE_DIR}')
        return None

    frames: List[pd.DataFrame] = []

    def _find_cols(cols):
        ts = None; gv = None; units = None
        for c in cols:
            cl = str(c).strip().lower()
            if ts is None and cl.startswith('timestamp'):
                ts = c
            if gv is None and 'glucose value' in cl:
                gv = c
                if 'mg/dl' in cl:
                    units = 'mgdl'
                elif 'mmol' in cl:
                    units = 'mmol'
        return ts, gv, units

    if verbose:
        print(f'[glucose] Walking: {GLUCOSE_DIR}')

    file_count = 0
    for root, _, files in os.walk(GLUCOSE_DIR):
        for fn in files:
            if not fn.lower().endswith('.csv'):
                continue
            if 'clarity_export' not in fn.lower():
                continue

            file_count += 1
            full = os.path.join(root, fn)
            print(f'[glucose] Found file: {full}') if verbose else None

            # PID = 3 digits before Sonic/SonicStudyVNS
            m = re.search(r'_(\d{3})_(?:SonicStudyVNS|Sonic)(?:_|$)', fn, flags=re.IGNORECASE)
            if not m:
                if verbose:
                    print(f"[glucose]   !! Could not parse PID from filename: {fn}")
                continue
            pid = str(int(m.group(1)))  # '011' -> '11'
            if verbose:
                print(f"[glucose]   PID from filename: {pid}")

            try:
                raw = pd.read_csv(full)
            except Exception as exc:
                if verbose:
                    print(f"[glucose]   !! Failed to read CSV: {exc}")
                continue

            if verbose:
                print(f"[glucose]   Columns: {list(raw.columns)}")

            ts_col, gv_col, units = _find_cols(list(raw.columns))
            if ts_col is None or gv_col is None:
                if verbose:
                    print(f"[glucose]   !! Missing timestamp/glucose columns; saw {list(raw.columns)[:8]} ...")
                continue

            if verbose:
                print(f"[glucose]   Using ts='{ts_col}', gv='{gv_col}', units={units or 'mmol'}")

            df = raw[[ts_col, gv_col]].copy()
            before = len(df)

            df.rename(columns={ts_col: 'Timestamp', gv_col: 'Glucose (mmol/L)'}, inplace=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Glucose (mmol/L)'] = pd.to_numeric(df['Glucose (mmol/L)'], errors='coerce')

            # Convert mg/dL → mmol/L
            if units == 'mgdl':
                df['Glucose (mmol/L)'] = df['Glucose (mmol/L)'] / 18.0

            df.dropna(subset=['Timestamp', 'Glucose (mmol/L)'], inplace=True)
            after = len(df)

            if verbose:
                print(f"[glucose]   Rows: raw={before}, clean={after}")
                if after:
                    print(f"[glucose]   Range: {df['Timestamp'].min()}  →  {df['Timestamp'].max()}")

            if df.empty:
                continue

            df['PID'] = pid
            frames.append(df)

    if verbose and file_count == 0:
        print("[glucose] No matching 'Clarity_Export*.csv' files found.")

    if not frames:
        if verbose:
            print('[glucose] No valid glucose rows across files.')
        return None

    out = pd.concat(frames, ignore_index=True)
    out.sort_values('Timestamp', inplace=True)

    if verbose:
        print(f"[glucose] TOTAL rows: {len(out)}  PIDs: {sorted(out['PID'].unique(), key=int)}")
        rng = (out['Timestamp'].min(), out['Timestamp'].max())
        print(f"[glucose] Overall time range: {rng[0]}  →  {rng[1]}")
        print(f"[glucose] Final columns: {list(out.columns)}")

    return out[['Timestamp', 'Glucose (mmol/L)', 'PID']]



def load_visits(verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load visit datetimes from Data/Visits/VisitDatesTimes.xlsx.

    Pairs any '...Date' column with the next '...Time' column.
    - Dates parsed with dayfirst=True (e.g., 05/11/2025 = 5 Nov 2025).
    - Times handled if string, Timestamp, datetime.time, or Excel serial (float).

    Returns: ['PID', 'Visit Time'] (Timestamp).
    """
    if not os.path.exists(VISITS_PATH):
        if verbose:
            print(f"[visits] Visits file not found: {VISITS_PATH}")
        return None

    try:
        df = pd.read_excel(VISITS_PATH)
    except Exception as exc:
        if verbose:
            print(f"[visits] Error reading {VISITS_PATH}: {exc}")
        return None

    # --- find participant column ---
    pid_col = None
    for c in df.columns:
        cl = str(c).lower()
        if 'participant' in cl and 'id' in cl:
            pid_col = c; break
    if pid_col is None:
        for c in df.columns:
            if 'participant' in str(c).lower():
                pid_col = c; break
    if pid_col is None:
        pid_col = df.columns[0]

    # --- pair (date, time) columns left->right ---
    cols = list(df.columns)
    pairs = []
    i = 0
    while i < len(cols):
        c = cols[i]
        if 'date' in str(c).lower():
            tcol = None
            for j in range(i + 1, len(cols)):
                if 'time' in str(cols[j]).lower():
                    tcol = cols[j]; break
            if tcol is not None:
                pairs.append((c, tcol))
                i = j + 1
                continue
        i += 1
    if verbose:
        print(f"[visits] Candidate pairs: {pairs}")
    if not pairs:
        if verbose:
            print("[visits] No (date, time) column pairs found.")
        return None

    def _parse_time_cell(v):
        """Return (h, m, s) from many possible time representations."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None

        # pandas Timestamp / python datetime
        if isinstance(v, (pd.Timestamp, dt.datetime)):
            t = v.to_pydatetime().time() if isinstance(v, pd.Timestamp) else v.time()
            return (t.hour, t.minute, t.second)

        # explicit time object
        if isinstance(v, dt.time):
            return (v.hour, v.minute, v.second)

        # time string like '09:00' or '09:00:00'
        if isinstance(v, str):
            tt = pd.to_datetime(v, errors='coerce')
            if pd.isna(tt):
                # try timedelta parsing e.g. '00:20:00'
                try:
                    td = pd.to_timedelta(v)
                    total = int(td.total_seconds())
                    return (total // 3600, (total % 3600) // 60, total % 60)
                except Exception:
                    return None
            t = tt.to_pydatetime().time()
            return (t.hour, t.minute, t.second)

        # Excel serial time as fraction of day (float)
        if isinstance(v, (int, float)) and not pd.isna(v):
            if 0 <= float(v) < 2:  # 0..1 is time-of-day; sometimes slightly >1
                secs = int(round(float(v) * 24 * 3600))
                return (secs // 3600, (secs % 3600) // 60, secs % 60)

        return None


    records = []
    for idx, row in df.iterrows():
        pid = _pid_from_pcode(row.get(pid_col))
        if pid is None:
            continue
        for dcol, tcol in pairs:
            d_raw = row.get(dcol, None)
            t_raw = row.get(tcol, None)
            # --- key change: dayfirst=True for dates ---
            d = pd.to_datetime(d_raw, errors='coerce', dayfirst=True)
            t_parts = _parse_time_cell(t_raw)
            if pd.isna(d) or t_parts is None:
                continue
            h, m, s = t_parts
            visit_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                    hour=h, minute=m, second=s)
            records.append({'PID': str(pid), 'Visit Time': visit_ts})

    if not records:
        if verbose:
            print("[visits] No visit rows parsed after cleaning.")
        return None

    out = pd.DataFrame.from_records(records).sort_values('Visit Time').reset_index(drop=True)

    if verbose:
        print(f"[visits] Parsed {len(out)} visits; PIDs={sorted(out['PID'].unique(), key=int)}")
        # show a few per PID for sanity
        for pid in sorted(out['PID'].unique(), key=int):
            sub = out[out['PID'] == pid]
            print(f"[visits]   PID {pid}: {len(sub)} visits; first={sub['Visit Time'].iloc[0]}")

    return out

def preview_cols(dfs, key):
    return list(dfs[key].columns) if key in dfs else []
