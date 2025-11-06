# new_dashboard.py
import os
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
from PyQt5.QtWidgets import (
    QWidget, QLabel, QComboBox, QPushButton, QDateEdit,
    QTabWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QSizePolicy,
    QTextEdit, QSplitter
)
from PyQt5.QtCore import QDate, Qt

from load_data import load_all_csvs, load_glucose_csvs, load_visits, DATA_DIR

DEBUG = True
def _dbg(*args):
    if DEBUG:
        print("[dash]", *args)

FIELD_MAP = {
    # Daily
    "dailySteps_merged":                   ("ActivityDay",       "StepTotal"),
    "dailyCalories_merged":                ("ActivityDay",       "Calories"),
    "dailyActivity_merged":                ("ActivityDate",      "Calories"),
    "dailyFitbitActiveZoneMinutes_merged": ("Date",              "FatBurnActiveZoneMinutes"),
    "dailyIntensities_merged":             ("ActivityDay",       "LightlyActiveMinutes"),
    "dailyCardioFitnessScore_merged":      ("DateTime",          "VO2Max"),
    "fitbitDailyHRV_merged":               ("SleepDay",          "DailyRMSSD"),
    "fitbitDailySpO2_merged":              ("SleepDay",          "AverageSpO2"),
    "fitbitSkinTemperature_merged":        ("SleepDay",          "NightlyRelative"),
    "fitbitBreathingRate_merged":          ("SleepDay",          "AvgBreathsPerMinute"),
    "sleepDay_merged":                     ("SleepDay",          "TotalMinutesAsleep"),
    "sleepStagesDay_merged":               ("SleepDay",          "TotalMinutesDeep"),

    # Hourly
    "hourlySteps_merged":                  ("ActivityHour",      "StepTotal"),
    "hourlyCalories_merged":               ("ActivityHour",      "Calories"),
    "hourlyIntensities_merged":            ("ActivityHour",      "TotalIntensity"),
    "heartrate_15min_merged":              ("Time",              "Value"),

    # Minute
    "heartrate_1min_merged":               ("Time",              "Value"),
    "minuteCaloriesNarrow_merged":         ("ActivityMinute",    "Calories"),
    "minuteStepsNarrow_merged":            ("ActivityMinute",    "Steps"),
    "fitbitMinuteSpO2_merged":             ("Time",              "SpO2"),

    # Multi-scale
    "fitbit5MinuteHRV_merged":             ("Time",              "RMSSD"),
}

METRIC_BY_SCALE = {
    "Daily":   [k for k in FIELD_MAP if k.startswith("daily") or k.startswith("fitbitDaily") or k.startswith("fitbitBreathingRate") or k.startswith("sleep")],
    "Hourly":  [k for k in FIELD_MAP if k.startswith("hourly") or k == "heartrate_15min_merged" or k == "fitbit5MinuteHRV_merged"],
    "Minute":  [k for k in FIELD_MAP if k.startswith("minute") or k in ("heartrate_1min_merged", "fitbitMinuteSpO2_merged", "fitbit5MinuteHRV_merged")],
}


class FitbitDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fitbit Data Dashboard")
        self.resize(1300, 900)

        self.dfs = load_all_csvs()
        self.glucose_df = load_glucose_csvs(verbose=False)  # Timestamp, Glucose (mmol/L), PID
        self.visits_df  = load_visits(verbose=False)

        if self.glucose_df is None or self.glucose_df.empty:
            _dbg("Glucose DF: None or empty")
        else:
            try:
                pids = sorted(self.glucose_df["PID"].astype(str).unique(),
                              key=lambda x: int(x) if x.isdigit() else x)
            except Exception:
                pids = list(self.glucose_df["PID"].astype(str).unique())
            _dbg(f"Glucose DF rows={len(self.glucose_df)} PIDs={pids} "
                 f"range={self.glucose_df['Timestamp'].min()} → {self.glucose_df['Timestamp'].max()}")

        # Notes state
        self.notes_path = os.path.join(DATA_DIR, "ParticipantNotes.csv")
        if os.path.exists(self.notes_path):
            try:
                self.notes_df = pd.read_csv(self.notes_path, dtype={"PID": str})
            except Exception:
                self.notes_df = pd.DataFrame(columns=["PID", "Notes", "Updated"])
        else:
            self.notes_df = pd.DataFrame(columns=["PID", "Notes", "Updated"])

        self._build_ui()
        self._set_default_view()
        self._update_plots("Daily")

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        left = QWidget()
        left_v = QVBoxLayout(left)

        self.participant_dropdown = QComboBox()
        self.participant_dropdown.addItem("All Participants", userData=None)

        pids = set()
        for df in self.dfs.values():
            if "PID" in df.columns:
                pids |= set(df["PID"].dropna().astype(str).unique())
        if self.glucose_df is not None and "PID" in self.glucose_df.columns:
            pids |= set(self.glucose_df["PID"].dropna().astype(str).unique())
        if self.visits_df is not None and "PID" in self.visits_df.columns:
            pids |= set(self.visits_df["PID"].dropna().astype(str).unique())

        for pid in sorted(pids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            self.participant_dropdown.addItem(f"Participant {pid}", userData=str(pid))
        self.participant_dropdown.currentIndexChanged.connect(self._on_participant_change)
        left_v.addWidget(self.participant_dropdown)

        self.tabs = QTabWidget()
        self.tab_widgets = {}
        for scale in ("Daily", "Hourly", "Minute"):
            tab = self._make_tab(scale)
            self.tabs.addTab(tab, scale)
            self.tab_widgets[scale] = tab
        left_v.addWidget(self.tabs)

        self.splitter.addWidget(left)

        # Notes panel
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(8, 8, 8, 8)
        hdr = QHBoxLayout()
        self.notes_title = QLabel("Notes"); self.notes_title.setStyleSheet("font-weight: 600;")
        self.collapse_btn = QPushButton("⟨⟨"); self.collapse_btn.setFixedWidth(32)
        self.collapse_btn.setToolTip("Collapse/Expand Notes panel")
        self.collapse_btn.clicked.connect(self._toggle_notes_panel)
        hdr.addWidget(self.notes_title); hdr.addStretch(1); hdr.addWidget(self.collapse_btn)
        rv.addLayout(hdr)
        self.notes_edit = QTextEdit(); self.notes_edit.setPlaceholderText("Write notes for this participant…")
        rv.addWidget(self.notes_edit, 1)
        self.save_notes_btn = QPushButton("Save"); self.save_notes_btn.clicked.connect(self._save_notes)
        rv.addWidget(self.save_notes_btn, 0, alignment=Qt.AlignRight)
        self.splitter.addWidget(right)
        self.splitter.setSizes([1000, 300])

        self._on_participant_change()

    def _compactize(self, w, maxw):
        w.setMaximumWidth(maxw)
        sp = w.sizePolicy(); sp.setHorizontalPolicy(QSizePolicy.Fixed); w.setSizePolicy(sp)

    def _make_tab(self, scale):
        tab = QWidget()
        vbox = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        metric_dropdown = QComboBox(); metric_dropdown.addItems(METRIC_BY_SCALE[scale]); self._compactize(metric_dropdown, 280)
        toolbar.addWidget(metric_dropdown)

        controls = {}
        if scale == "Daily":
            sdw = QDateEdit(QDate(2025, 10, 1)); edw = QDateEdit(QDate(2025, 10, 31))
            for w in (sdw, edw): w.setCalendarPopup(True); self._compactize(w, 130)
            gs = QComboBox(); gs.addItems(["30 minutes", "1 hour", "4 hours", "12 hours", "1 day"]); self._compactize(gs, 150)
            controls.update(start=sdw, end=edw, g_smooth=gs)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Start:")); toolbar.addWidget(sdw)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End:"));   toolbar.addWidget(edw)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Glucose smoothing:")); toolbar.addWidget(gs)

        elif scale == "Hourly":
            datew = QDateEdit(QDate(2025, 10, 15)); datew.setCalendarPopup(True); self._compactize(datew, 130)
            sh = QSpinBox(); sh.setRange(0,23); sh.setValue(0); self._compactize(sh, 60)
            eh = QSpinBox(); eh.setRange(0,23); eh.setValue(23); self._compactize(eh, 60)
            gs = QComboBox(); gs.addItems(["5 minutes", "10 minutes", "30 minutes", "1 hour"]); self._compactize(gs,150)
            controls.update(date=datew, start_hour=sh, end_hour=eh, g_smooth=gs)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Date:")); toolbar.addWidget(datew)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Start Hour:")); toolbar.addWidget(sh)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End Hour:"));   toolbar.addWidget(eh)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Glucose smoothing:")); toolbar.addWidget(gs)

        else:
            datew = QDateEdit(QDate(2025, 10, 15)); datew.setCalendarPopup(True); self._compactize(datew, 130)
            hr = QSpinBox(); hr.setRange(0,23); hr.setValue(12); self._compactize(hr, 60)
            sm = QSpinBox(); sm.setRange(0,59); sm.setValue(0); self._compactize(sm, 60)
            em = QSpinBox(); em.setRange(0,59); em.setValue(59); self._compactize(em, 60)
            controls.update(date=datew, hour=hr, start_minute=sm, end_minute=em)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Date:")); toolbar.addWidget(datew)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Hour:")); toolbar.addWidget(hr)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Start Min:")); toolbar.addWidget(sm)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End Min:"));  toolbar.addWidget(em)

        toolbar.addStretch(1)
        vbox.addLayout(toolbar)

        update_btn = QPushButton("Update Plots")
        update_btn.clicked.connect(lambda _, s=scale: self._update_plots(s))
        vbox.addWidget(update_btn)

        top_fig = plt.figure(figsize=(8,4)); top_canvas = FigureCanvas(top_fig); vbox.addWidget(top_canvas)
        bot_fig = plt.figure(figsize=(8,3)); bot_canvas = FigureCanvas(bot_fig); vbox.addWidget(bot_canvas)

        tab.metric_dropdown = metric_dropdown
        tab.controls = controls
        tab.top_fig, tab.top_canvas = top_fig, top_canvas
        tab.bot_fig, tab.bot_canvas = bot_fig, bot_canvas
        return tab

    def _set_default_view(self):
        idx = self.participant_dropdown.findData("1")
        if idx != -1: self.participant_dropdown.setCurrentIndex(idx)
        daily_tab = self.tab_widgets["Daily"]
        daily_tab.controls['start'].setDate(QDate(2025, 10, 1))
        daily_tab.controls['end'].setDate(QDate(2025, 10, 31))
        self.tabs.setCurrentIndex(0)

    # Notes panel
    def _toggle_notes_panel(self):
        sizes = self.splitter.sizes()
        if sizes[-1] == 0:
            self.splitter.setSizes([max(1, sizes[0]-300), 300]); self.collapse_btn.setText("⟨⟨")
        else:
            self.splitter.setSizes([sum(sizes), 0]); self.collapse_btn.setText("⟩⟩")

    def _on_participant_change(self):
        pid = self.participant_dropdown.currentData()
        right_size = self.splitter.sizes()[-1]
        if pid is None:
            if right_size != 0: self.splitter.setSizes([sum(self.splitter.sizes()), 0])
            self.collapse_btn.setEnabled(False); self.save_notes_btn.setEnabled(False)
            self.notes_edit.setPlainText(""); self.notes_title.setText("Notes")
        else:
            if right_size == 0: self.splitter.setSizes([1000, 300])
            self.collapse_btn.setEnabled(True); self.save_notes_btn.setEnabled(True)
            self.notes_title.setText(f"Notes — Participant {pid}")
            txt = ""
            if not self.notes_df.empty:
                hit = self.notes_df[self.notes_df["PID"].astype(str) == str(pid)]
                if not hit.empty and isinstance(hit.iloc[0].get("Notes",""), str):
                    txt = hit.iloc[0]["Notes"]
            if self.notes_edit.toPlainText() != txt: self.notes_edit.setPlainText(txt)

    def _save_notes(self):
        pid = self.participant_dropdown.currentData()
        if pid is None: return
        text = self.notes_edit.toPlainText(); now = pd.Timestamp.now().isoformat(timespec="seconds")
        if self.notes_df.empty:
            self.notes_df = pd.DataFrame([{"PID": str(pid), "Notes": text, "Updated": now}])
        else:
            mask = self.notes_df["PID"].astype(str) == str(pid)
            if mask.any():
                self.notes_df.loc[mask, ["Notes","Updated"]] = [text, now]
            else:
                self.notes_df = pd.concat([self.notes_df, pd.DataFrame([{"PID": str(pid), "Notes": text, "Updated": now}])], ignore_index=True)
        self.notes_df["PID"] = self.notes_df["PID"].astype(str)
        cols = ["PID","Notes","Updated"]
        for c in cols:
            if c not in self.notes_df.columns: self.notes_df[c] = ""
        self.notes_df[cols].to_csv(self.notes_path, index=False)

    # helpers
    def _parse_smooth_selection(self, text: str) -> str:
        mapping = {"5 minutes":"5min","10 minutes":"10min","30 minutes":"30min","1 hour":"1h","4 hours":"4h","12 hours":"12h","1 day":"1d"}
        return mapping.get(text.lower().strip(), "5min")

    @staticmethod
    def _stats_label(ax, values):
        s = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
        text = "Mean: –\nVariance: –" if s.empty else f"Mean: {s.mean():.3g}\nVariance: {s.var(ddof=0 if len(s)==1 else 1):.3g}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    @staticmethod
    def _day_tick_interval(start_dt, end_dt, target=8):
        days = max(1, (pd.to_datetime(end_dt).date() - pd.to_datetime(start_dt).date()).days + 1)
        return max(1, int(np.ceil(days / target)))

    # ---------------- DATA & PLOTS ----------------
    def _update_plots(self, scale):
        tab = self.tab_widgets[scale]
        metric = tab.metric_dropdown.currentText()

        fig = tab.top_fig; fig.clear(); ax = fig.add_subplot(111)
        bfig = tab.bot_fig; bfig.clear(); bax = bfig.add_subplot(111)

        if metric not in self.dfs:
            ax.text(0.5, 0.5, f"'{metric}.csv' not loaded or missing in Data/", ha='center', va='center')
            ax.grid(True); tab.top_canvas.draw()
            bax.grid(True); tab.bot_canvas.draw(); return

        df = self.dfs[metric].copy()
        time_col, val_col = FIELD_MAP[metric]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[val_col]  = pd.to_numeric(df[val_col], errors='coerce')
        df.dropna(subset=[time_col, val_col], inplace=True)

        sel_pid = self.participant_dropdown.currentData()
        if sel_pid is not None and "PID" in df.columns:
            df = df[df["PID"].astype(str) == str(sel_pid)]

        # Build time window
        if scale == "Daily":
            sd = tab.controls['start'].date().toPyDate(); ed = tab.controls['end'].date().toPyDate()
            start_dt = datetime.combine(sd, time.min); end_dt = datetime.combine(ed, time.max)
            df = df[df[time_col].between(start_dt, end_dt)].copy()
            df['x'] = df[time_col].dt.normalize()

        elif scale == "Hourly":
            date_only = tab.controls['date'].date().toPyDate()
            sh = tab.controls['start_hour'].value(); eh = tab.controls['end_hour'].value()
            start_dt = datetime.combine(date_only, time(sh))
            end_dt   = datetime.combine(date_only, time(eh, 59, 59))
            df = df[df[time_col].between(start_dt, end_dt)].copy()
            df['x'] = df[time_col].dt.floor('h')

        else:  # Minute
            date_only = tab.controls['date'].date().toPyDate()
            hr = tab.controls['hour'].value(); sm = tab.controls['start_minute'].value(); em = tab.controls['end_minute'].value()
            start_dt = datetime.combine(date_only, time(hr, sm))
            end_dt   = datetime.combine(date_only, time(hr, em, 59))
            df = df[df[time_col].between(start_dt, end_dt)].copy()
            df['x'] = df[time_col].dt.floor('min')

        _dbg(f"Scale={scale}  PID={sel_pid}  Window={start_dt} → {end_dt}")

        # ----- TOP plot -----
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            metric_vals_for_stats = []
        else:
            if sel_pid is None and "PID" in df.columns:
                agg = df.groupby(['x','PID'])[val_col].mean().unstack()
                agg.plot(ax=ax, marker='o', linewidth=1.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), title="Participant", loc='upper right')
                metric_vals_for_stats = agg.stack().values
            else:
                agg = df.groupby('x')[val_col].mean()
                agg.plot(ax=ax, marker='o', linewidth=1.2)
                metric_vals_for_stats = agg.values

        ax.set_ylabel(val_col); ax.set_xlabel("")

        # Axes formatting
        if scale == "Daily":
            interval = self._day_tick_interval(start_dt, end_dt)
            start_day = pd.to_datetime(start_dt).normalize()
            end_day   = pd.to_datetime(end_dt).normalize()
            ticks = pd.date_range(start=start_day, end=end_day, freq=f'{interval}D')
            if ticks[0] != start_day: ticks = ticks.insert(0, start_day)
            if ticks[-1] != end_day:  ticks = ticks.append(pd.DatetimeIndex([end_day]))
            ax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt)); ax.margins(x=0)
            ax.set_xticks([t.to_pydatetime() for t in ticks])
            def _fmt_end_blank(x, _pos=None, end_norm=end_day):
                d = pd.to_datetime(mdates.num2date(x)).normalize()
                return "" if d == end_norm else d.strftime('%d/%m')
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_end_blank))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        elif scale == "Hourly":
            # NEW: extend ticks one hour past the end hour
            start_h = pd.to_datetime(start_dt).replace(minute=0, second=0, microsecond=0)
            end_h   = pd.to_datetime(end_dt).replace(minute=0, second=0, microsecond=0)
            end_h_plus = end_h + timedelta(hours=1)
            ticks = pd.date_range(start=start_h, end=end_h_plus, freq="1h")
            ax.set_xlim(pd.to_datetime(start_dt), end_h_plus)
            # Label last midnight tick as "24" when it rolls into next day
            labels = []
            for t in ticks:
                if t.date() > start_h.date() and t.hour == 0:
                    labels.append("24")
                else:
                    labels.append(str(t.hour))
            ax.set_xticks([t.to_pydatetime() for t in ticks])
            ax.set_xticklabels(labels)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        else:  # Minute
            ax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, which='both', axis='x')

        ax.grid(True); self._stats_label(ax, metric_vals_for_stats); tab.top_canvas.draw()

        # ----- BOTTOM (Glucose) -----
        bfig = tab.bot_fig; bfig.clear(); bax = bfig.add_subplot(111)
        ts_col = 'Timestamp'; gv_col = 'Glucose (mmol/L)'; plotted_y_values = []

        if self.glucose_df is None or self.glucose_df.empty:
            bax.text(0.5, 0.5, 'No glucose data', ha='center', va='center'); _dbg("Glucose DF empty/None")
        else:
            g = self.glucose_df.copy()
            g[ts_col] = pd.to_datetime(g[ts_col], errors='coerce')
            g[gv_col] = pd.to_numeric(g[gv_col], errors='coerce')
            g.dropna(subset=[ts_col, gv_col], inplace=True)
            _dbg(f"Glucose total rows={len(g)}; time range={g[ts_col].min()} → {g[ts_col].max()}")
            g = g[(g[ts_col] >= start_dt) & (g[ts_col] <= end_dt)]
            _dbg(f"Glucose after time filter: {len(self.glucose_df)} → {len(g)}")
            if sel_pid is not None and "PID" in g.columns:
                before = len(g); g = g[g["PID"].astype(str) == str(sel_pid)]
                _dbg(f"Glucose after PID filter:  {before} → {len(g)} (PID={sel_pid})")

            if g.empty:
                bax.text(0.5, 0.5, 'No glucose data', ha='center', va='center'); _dbg("Glucose: NO ROWS after filters")
            else:
                rule = None
                if scale in ("Daily","Hourly"):
                    rule = self._parse_smooth_selection(tab.controls['g_smooth'].currentText())
                _dbg(f"Glucose smoothing rule={rule}")

                def plot_glucose_df(df_g, label=None):
                    if df_g.empty: return
                    bax.plot(df_g[ts_col], df_g[gv_col], marker='o', linewidth=1.2, label=label)
                    plotted_y_values.extend(df_g[gv_col].values.tolist())

                if sel_pid is None and "PID" in g.columns:
                    for pid, sub in g.groupby("PID"):
                        sub = sub.sort_values(ts_col)
                        if rule:
                            if scale == "Daily" and rule == "1d":
                                sub = (sub.set_index(ts_col)[gv_col]
                                         .resample('1d', label='left', closed='left').mean().reset_index())
                                sub = sub[(sub[ts_col] >= pd.to_datetime(start_dt).normalize()) &
                                          (sub[ts_col] <= pd.to_datetime(end_dt).normalize())]
                            else:
                                sub = (sub.set_index(ts_col)[gv_col].resample(rule).mean().reset_index())
                        plot_glucose_df(sub, label=str(pid))
                    bax.legend(title="Participant", loc='upper right')
                else:
                    g = g.sort_values(ts_col)
                    if rule:
                        if scale == "Daily" and rule == "1d":
                            g = (g.set_index(ts_col)[gv_col]
                                   .resample('1d', label='left', closed='left').mean().reset_index())
                            g = g[(g[ts_col] >= pd.to_datetime(start_dt).normalize()) &
                                  (g[ts_col] <= pd.to_datetime(end_dt).normalize())]
                        else:
                            g = (g.set_index(ts_col)[gv_col].resample(rule).mean().reset_index())
                    plot_glucose_df(g)

                if plotted_y_values:
                    y = np.array(plotted_y_values, dtype=float); y = y[np.isfinite(y)]
                    if y.size > 0 and np.nanmin(y) != np.nanmax(y):
                        pad = 0.05 * (np.nanmax(y) - np.nanmin(y))
                        bax.set_ylim(np.nanmin(y)-pad, np.nanmax(y)+pad)

        # ----- Visit lines (absolute order coloring) -----
        if self.visits_df is not None and sel_pid is not None:
            all_vis = (self.visits_df[self.visits_df['PID'].astype(str)==str(sel_pid)]
                       .sort_values('Visit Time')['Visit Time'].tolist())
            order_map = {pd.to_datetime(ts): i for i, ts in enumerate(all_vis)}
            in_window = [ts for ts in all_vis if start_dt <= pd.to_datetime(ts) <= end_dt]
            if in_window:
                _dbg(f"Visit lines (window): {len(in_window)}")
            colors_cycle = ['black','red','orange','black']
            for ts in in_window:
                idx = order_map[pd.to_datetime(ts)]
                color = colors_cycle[idx] if idx < len(colors_cycle) else 'gray'
                bax.axvline(pd.to_datetime(ts), color=color, linestyle='--', linewidth=1.2, alpha=0.95)

        # ----- Bottom axis ticks -----
        if scale == "Hourly":
            # NEW: extend one hour beyond end hour (and label 24 at midnight)
            start_h = pd.to_datetime(start_dt).replace(minute=0, second=0, microsecond=0)
            end_h   = pd.to_datetime(end_dt).replace(minute=0, second=0, microsecond=0)
            end_h_plus = end_h + timedelta(hours=1)
            ticks = pd.date_range(start=start_h, end=end_h_plus, freq="1h")
            bax.set_xlim(pd.to_datetime(start_dt), end_h_plus)
            labels = []
            for t in ticks:
                if t.date() > start_h.date() and t.hour == 0:
                    labels.append("24")
                else:
                    labels.append(str(t.hour))
            bax.set_xticks([t.to_pydatetime() for t in ticks])
            bax.set_xticklabels(labels)
            bax.xaxis.set_minor_locator(NullLocator())

        elif scale == "Minute":
            bax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            bax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            bax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            bax.grid(True, which='both', axis='x')
        else:
            bax.set_xlim(ax.get_xlim())
            bax.set_xticks(ax.get_xticks())
            try:
                bax.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())
                bax.xaxis.set_minor_locator(ax.xaxis.get_minor_locator())
            except Exception:
                pass

        bax.set_ylabel('Glucose (mmol/L)'); bax.grid(True); tab.bot_canvas.draw()
