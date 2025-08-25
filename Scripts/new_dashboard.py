from datetime import datetime, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
from PyQt5.QtWidgets import (
    QWidget, QLabel, QComboBox, QPushButton, QDateEdit,
    QTabWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QSizePolicy
)
from PyQt5.QtCore import QDate, Qt

from load_data import load_all_csvs, load_glucose_csvs, load_stimulation

# Map each dataset to its timestamp column and default value column
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

    # Multi-scale (Hourly + Minute)
    "fitbit5MinuteHRV_merged":             ("Time",              "RMSSD"),
}

# Dropdown mapping for each view
METRIC_BY_SCALE = {
    "Daily":   [k for k in FIELD_MAP if k.startswith("daily") or k.startswith("fitbitDaily") or k.startswith("fitbitBreathingRate") or k.startswith("sleep")],
    "Hourly":  [k for k in FIELD_MAP if k.startswith("hourly") or k == "heartrate_15min_merged" or k == "fitbit5MinuteHRV_merged"],
    "Minute":  [k for k in FIELD_MAP if k.startswith("minute") or k in ("heartrate_1min_merged", "fitbitMinuteSpO2_merged", "fitbit5MinuteHRV_merged")],
}


class FitbitDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fitbit Data Dashboard")
        self.resize(1200, 900)

        self.dfs = load_all_csvs()
        self.glucose_df = load_glucose_csvs(verbose=False)
        self.stim_df = load_stimulation(verbose=False)  # <<— NEW

        self._build_ui()
        self._update_plots("Daily")

    # ---------------- UI ----------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Participant selector (compact, no label above)
        self.participant_dropdown = QComboBox()
        self.participant_dropdown.addItem("All Participants", userData=None)
        pids = set()
        for df in self.dfs.values():
            if "PID" in df.columns:
                pids |= set(df["PID"].dropna().astype(str).unique())
        if self.glucose_df is not None and "PID" in self.glucose_df.columns:
            pids |= set(self.glucose_df["PID"].dropna().astype(str).unique())
        if self.stim_df is not None and "PID" in self.stim_df.columns:
            pids |= set(self.stim_df["PID"].dropna().astype(str).unique())
        for pid in sorted(pids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
            self.participant_dropdown.addItem(f"Participant {pid}", userData=str(pid))
        layout.addWidget(self.participant_dropdown)

        # Tabs
        self.tabs = QTabWidget()
        self.tab_widgets = {}
        for scale in ("Daily", "Hourly", "Minute"):
            tab = self._make_tab(scale)
            self.tabs.addTab(tab, scale)
            self.tab_widgets[scale] = tab
        layout.addWidget(self.tabs)

    # builders
    def _compactize(self, w, maxw):
        w.setMaximumWidth(maxw)
        sp = w.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Fixed)
        w.setSizePolicy(sp)

    def _make_tab(self, scale):
        tab = QWidget()
        vbox = QVBoxLayout(tab)

        # Single toolbar row: metric + date/time + (optional) smoothing
        toolbar = QHBoxLayout()

        metric_dropdown = QComboBox()
        metric_dropdown.addItems(METRIC_BY_SCALE[scale])
        self._compactize(metric_dropdown, 280)
        toolbar.addWidget(metric_dropdown)

        controls = {}
        # Defaults
        if scale == "Daily":
            sd = QDate(2025, 6, 18)
            ed = QDate(2025, 6, 21)
            sdw = QDateEdit(sd); sdw.setCalendarPopup(True); self._compactize(sdw, 130)
            edw = QDateEdit(ed); edw.setCalendarPopup(True); self._compactize(edw, 130)
            gs = QComboBox(); self._compactize(gs, 150)
            gs.addItems(["30 minutes", "1 hour", "4 hours", "12 hours", "1 day"])
            controls.update(start=sdw, end=edw, g_smooth=gs)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Start:")); toolbar.addWidget(sdw)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End:"));   toolbar.addWidget(edw)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Glucose smoothing:")); toolbar.addWidget(gs)

        elif scale == "Hourly":
            date = QDate(2025, 6, 19)
            datew = QDateEdit(date); datew.setCalendarPopup(True); self._compactize(datew, 130)
            sh = QSpinBox(); sh.setRange(0,23); sh.setValue(0); self._compactize(sh, 60)
            eh = QSpinBox(); eh.setRange(0,23); eh.setValue(23); self._compactize(eh, 60)
            gs = QComboBox(); self._compactize(gs, 150)
            gs.addItems(["5 minutes", "10 minutes", "30 minutes", "1 hour"])
            controls.update(date=datew, start_hour=sh, end_hour=eh, g_smooth=gs)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Date:")); toolbar.addWidget(datew)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Start Hour:")); toolbar.addWidget(sh)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End Hour:"));   toolbar.addWidget(eh)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Glucose smoothing:")); toolbar.addWidget(gs)

        else:  # Minute
            date = QDate(2025, 6, 19)
            datew = QDateEdit(date); datew.setCalendarPopup(True); self._compactize(datew, 130)
            hr = QSpinBox(); hr.setRange(0,23); hr.setValue(0); self._compactize(hr, 60)
            sm = QSpinBox(); sm.setRange(0,59); sm.setValue(0); self._compactize(sm, 60)
            em = QSpinBox(); em.setRange(0,59); em.setValue(59); self._compactize(em, 60)
            controls.update(date=datew, hour=hr, start_minute=sm, end_minute=em)
            toolbar.addSpacing(12); toolbar.addWidget(QLabel("Date:")); toolbar.addWidget(datew)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Hour:")); toolbar.addWidget(hr)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("Start Min:")); toolbar.addWidget(sm)
            toolbar.addSpacing(8);  toolbar.addWidget(QLabel("End Min:")); toolbar.addWidget(em)

        toolbar.addStretch(1)
        vbox.addLayout(toolbar)

        update_btn = QPushButton("Update Plots")
        update_btn.clicked.connect(lambda _, s=scale: self._update_plots(s))
        vbox.addWidget(update_btn)

        top_fig = plt.figure(figsize=(8,4))
        top_canvas = FigureCanvas(top_fig)
        vbox.addWidget(top_canvas)

        bot_fig = plt.figure(figsize=(8,3))
        bot_canvas = FigureCanvas(bot_fig)
        vbox.addWidget(bot_canvas)

        tab.metric_dropdown = metric_dropdown
        tab.controls = controls
        tab.top_fig, tab.top_canvas = top_fig, top_canvas
        tab.bot_fig, tab.bot_canvas = bot_fig, bot_canvas
        return tab

    # helpers 
    def _parse_smooth_selection(self, text: str) -> str:
        mapping = {
            "5 minutes": "5min",
            "10 minutes": "10min",
            "30 minutes": "30min",
            "1 hour": "1h",
            "4 hours": "4h",
            "12 hours": "12h",
            "1 day": "1d",
        }
        return mapping.get(text.lower().strip(), "5min")

    @staticmethod
    def _stats_label(ax, values):
        s = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
        if s.empty:
            text = "Mean: –\nVariance: –"
        else:
            mean = s.mean()
            var = s.var(ddof=0 if len(s) == 1 else 1)
            text = f"Mean: {mean:.3g}\nVariance: {var:.3g}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    @staticmethod
    def _day_tick_interval(start_dt, end_dt, target=8):
        days = max(1, (pd.to_datetime(end_dt).date() - pd.to_datetime(start_dt).date()).days + 1)
        return max(1, int(np.ceil(days / target)))

    @staticmethod
    def _hour_ticks(start_dt, end_dt, target_labels=8):
        start = pd.to_datetime(start_dt).replace(minute=0, second=0, microsecond=0)
        end   = pd.to_datetime(end_dt).replace(minute=0, second=0, microsecond=0)
        hours = int((end - start).total_seconds() // 3600) + 1
        step  = max(1, int(np.ceil(hours / target_labels)))
        ticks = pd.date_range(start=start, end=end, freq=f"{step}h")
        return ticks, step

    # DATA & PLOTS 
    def _update_plots(self, scale):
        tab = self.tab_widgets[scale]
        metric = tab.metric_dropdown.currentText()

        fig = tab.top_fig; fig.clear(); ax = fig.add_subplot(111)
        bfig = tab.bot_fig; bfig.clear(); bax = bfig.add_subplot(111)

        if metric not in self.dfs:
            ax.text(0.5, 0.5, f"'{metric}.csv' not loaded or missing in Data/", ha='center', va='center')
            ax.grid(True)
            tab.top_canvas.draw()

            bax.text(0.5, 0.5, 'Glucose panel disabled (metric missing)', ha='center', va='center')
            bax.grid(True)
            tab.bot_canvas.draw()
            return

        df = self.dfs[metric].copy()
        time_col, val_col = FIELD_MAP[metric]

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df.dropna(subset=[time_col, val_col], inplace=True)

        sel_pid = self.participant_dropdown.currentData()
        if sel_pid is not None and "PID" in df.columns:
            df = df[df["PID"].astype(str) == str(sel_pid)]

        if scale == "Daily":
            sd = tab.controls['start'].date().toPyDate(); ed = tab.controls['end'].date().toPyDate()
            start_dt = datetime.combine(sd, time.min)
            end_dt   = datetime.combine(ed, time.max)
            mask = df[time_col].between(start_dt, end_dt)
            df = df.loc[mask].copy()
            df['x'] = df[time_col].dt.normalize()  # exact day bins

        elif scale == "Hourly":
            date = tab.controls['date'].date().toPyDate()
            sh = tab.controls['start_hour'].value(); eh = tab.controls['end_hour'].value()
            start_dt = datetime.combine(date, time(sh))
            end_dt   = datetime.combine(date, time(eh, 59, 59))
            mask = df[time_col].between(start_dt, end_dt)
            df = df.loc[mask].copy()
            df['x'] = df[time_col].dt.floor('h')

        else:  # Minute
            date = tab.controls['date'].date().toPyDate()
            hr = tab.controls['hour'].value(); sm = tab.controls['start_minute'].value(); em = tab.controls['end_minute'].value()
            start_dt = datetime.combine(date, time(hr, sm))
            end_dt   = datetime.combine(date, time(hr, em, 59))
            mask = df[time_col].between(start_dt, end_dt)
            df = df.loc[mask].copy()
            df['x'] = df[time_col].dt.floor('min')

        # TOP PLOT: aggregate metric
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            metric_vals_for_stats = []
        else:
            if sel_pid is None and "PID" in df.columns:
                agg = df.groupby(['x', 'PID'])[val_col].mean().unstack()
                agg.plot(ax=ax, marker='o')
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, title="Participant", loc='upper right')
                metric_vals_for_stats = agg.stack().values
            else:
                agg = df.groupby('x')[val_col].mean()
                agg.plot(ax=ax, marker='o')
                metric_vals_for_stats = agg.values

        ax.set_ylabel(val_col)
        ax.set_xlabel("")

        # Axes formatting
        if scale == "Daily":
            interval = self._day_tick_interval(start_dt, end_dt)
            start_day = pd.to_datetime(start_dt).normalize()
            end_day   = pd.to_datetime(end_dt).normalize()
            ticks = pd.date_range(start=start_day, end=end_day, freq=f'{interval}D')
            if ticks[0] != start_day:
                ticks = ticks.insert(0, start_day)
            if ticks[-1] != end_day:
                ticks = ticks.append(pd.DatetimeIndex([end_day]))
            ax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            ax.margins(x=0)
            ax.set_xticks(ticks)
            def _fmt_end_blank(x, _pos=None, end_norm=end_day):
                d = pd.to_datetime(mdates.num2date(x)).normalize()
                return "" if d == end_norm else d.strftime('%d/%m')
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_end_blank))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        elif scale == "Hourly":
            hours = pd.date_range(start_dt, end_dt, freq="h")
            ax.set_xlim(hours[0], hours[-1])
            max_ticks = 8
            step = max(1, len(hours) // max_ticks)
            tick_positions = hours[::step]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(ts.hour) for ts in tick_positions])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        else:  # Minute
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            ax.grid(True, which='both', axis='x')
        ax.grid(True)

        # Metric stats label (upper-left)
        self._stats_label(ax, metric_vals_for_stats)
        tab.top_canvas.draw()

        # BOTTOM: Glucose plot 
        bfig = tab.bot_fig; bfig.clear(); bax = bfig.add_subplot(111)
        if self.glucose_df is None:
            bax.text(0.5, 0.5, 'No glucose data', ha='center', va='center')
            bax.grid(True)
            tab.bot_canvas.draw(); return

        ts_col = 'Timestamp (YYYY-MM-DDThh:mm:ss)'
        gv_col = 'Glucose Value (mmol/L)'

        g = self.glucose_df.copy()
        g[ts_col] = pd.to_datetime(g[ts_col], errors='coerce')
        g[gv_col] = pd.to_numeric(g[gv_col], errors='coerce')
        g.dropna(subset=[ts_col, gv_col], inplace=True)
        g = g[(g[ts_col] >= start_dt) & (g[ts_col] <= end_dt)]
        if sel_pid is not None and "PID" in g.columns:
            g = g[g["PID"].astype(str) == str(sel_pid)]

        rule = None
        if scale in ("Daily", "Hourly"):
            rule = self._parse_smooth_selection(tab.controls['g_smooth'].currentText())

        plotted_y_values = []

        def plot_glucose_df(df_g, label=None):
            if df_g.empty:
                return
            bax.plot(df_g[ts_col], df_g[gv_col], marker='o', label=label)
            plotted_y_values.extend(df_g[gv_col].values.tolist())

        if g.empty:
            bax.text(0.5, 0.5, 'No glucose data', ha='center', va='center')
        else:
            if sel_pid is None and "PID" in g.columns:
                for pid, sub in g.groupby("PID"):
                    sub = sub.sort_values(ts_col)
                    if rule:
                        if scale == "Daily" and rule == "1d":
                            sub = (sub.set_index(ts_col)[gv_col]
                                       .resample('1d', label='left', closed='left')
                                       .mean().reset_index())
                            sub = sub[(sub[ts_col] >= pd.to_datetime(start_dt).normalize()) &
                                      (sub[ts_col] <= pd.to_datetime(end_dt).normalize())]
                        else:
                            sub = (sub.set_index(ts_col)[gv_col]
                                       .resample(rule).mean()
                                       .reset_index())
                    plot_glucose_df(sub, label=str(pid))
                bax.legend(title="Participant", loc='upper right')
            else:
                g = g.sort_values(ts_col)
                if rule:
                    if scale == "Daily" and rule == "1d":
                        g = (g.set_index(ts_col)[gv_col]
                               .resample('1d', label='left', closed='left')
                               .mean().reset_index())
                        g = g[(g[ts_col] >= pd.to_datetime(start_dt).normalize()) &
                              (g[ts_col] <= pd.to_datetime(end_dt).normalize())]
                    else:
                        g = (g.set_index(ts_col)[gv_col]
                               .resample(rule).mean()
                               .reset_index())
                plot_glucose_df(g)

        # NEW: overlay stimulation start/end lines for the selected participant 
        if self.stim_df is not None and sel_pid is not None:
            stim = self.stim_df.copy()
            stim = stim[stim['PID'].astype(str) == str(sel_pid)]
            if not stim.empty:
                # keep intervals that intersect [start_dt, end_dt]
                stim = stim[(stim['Start Time'] <= end_dt) & (stim['End Time'] >= start_dt)]
                # clip to range (for visibility in minute/hour views)
                for _, row in stim.iterrows():
                    st = max(pd.to_datetime(row['Start Time']), pd.to_datetime(start_dt))
                    en = min(pd.to_datetime(row['End Time']),   pd.to_datetime(end_dt))
                    # vertical lines
                    bax.axvline(st, color='green', linestyle='--', linewidth=2, alpha=0.9)
                    bax.axvline(en, color='red', linestyle='--', linewidth=2, alpha=0.9)

        bax.set_ylabel('Glucose (mmol/L)')
        bax.set_yscale('linear')

        if plotted_y_values:
            y = np.array(plotted_y_values, dtype=float)
            y = y[np.isfinite(y)]
            if y.size > 0 and np.nanmin(y) != np.nanmax(y):
                pad = 0.05 * (np.nanmax(y) - np.nanmin(y))
                bax.set_ylim(np.nanmin(y) - pad, np.nanmax(y) + pad)

        # Axes formatting (match top)
        if scale == "Daily":
            bax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            bax.margins(x=0)
            bax.set_xticks(ax.get_xticks())
            end_day = pd.to_datetime(end_dt).normalize()
            def _fmt_end_blank_btm(x, _pos=None, end_norm=end_day):
                d = pd.to_datetime(mdates.num2date(x)).normalize()
                return "" if d == end_norm else d.strftime('%d/%m')
            bax.xaxis.set_major_formatter(FuncFormatter(_fmt_end_blank_btm))
            bax.xaxis.set_minor_locator(NullLocator())
            bax.set_title('')
        elif scale == "Hourly":
            hours = pd.date_range(start_dt, end_dt, freq="h")
            bax.set_xlim(hours[0], hours[-1])
            max_ticks = 8
            step = max(1, len(hours) // max_ticks)
            tick_positions = hours[::step]
            bax.set_xticks(tick_positions)
            bax.set_xticklabels([str(ts.hour) for ts in tick_positions])
            bax.xaxis.set_minor_locator(NullLocator())
        else:  # Minute
            bax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            bax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            bax.set_xlim(pd.to_datetime(start_dt), pd.to_datetime(end_dt))
            bax.grid(True, which='both', axis='x')

        bax.grid(True)
        # Glucose stats label (upper-left)
        self._stats_label(bax, plotted_y_values)

        tab.bot_canvas.draw()
