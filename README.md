# NeuMeDiC

# Fitbit and Glucose Data Dashboard

This is a PyQt5-based interactive dashboard for visualising Fitbit data, continuous glucose monitoring data, and non-invasive vagus nerve stimulation intervals, in the wider context of the NeuMeDiC project. It allows users to view individual or combined participant data across daily, hourly, or minute-level resolution.

## Features

- View daily, hourly, or minute-level Fitbit data
- Overlay Dexcom CGM glucose levels from Clarity exports
- Mark stimulation intervals with vertical lines on glucose plots:
- Automatic summary statistics for each plot
- Supports individual or all participant views

## Setup

- Necessary modules are listed in requirements.txt

```bash
pip install -r requirements.txt
```
- As of 25 Aug 2025, you can also just run:

```bash
pip install pandas numpy matplotlib PyQt5 word2number openpyxl
```
- Setting up a virtual environment at the top level is recommended

```bash
python3.7 -m venv venv
source venv/bin/activate
```

## Structure

```text
DASHBOARD/
├── Data/
│   ├── dailySteps_merged.csv
│   ├── dailyCalories_merged.csv
│   ├── ...
│   ├── Glucose/
│   │   └── Clarity_Export_One_Participant.csv
│   │   └── ...
│   └── Stimulation.xlsx
├── Scripts/
│   ├── load_data.py
│   ├── new_dashboard.py
│   └── main.py
├── README.md
├── requirements.txt
```
- Merged (i.e. all participants) fitbit data files must go directly in Data. Glucose exported data must go in Glucose inside Data. 
- Any CSV added to the Data folder will be automatically loaded.
The UI dropdowns are mapped via FIELD_MAP in new_dashboard.py, which is hardcoded. If a file name differs, you must update FIELD_MAP accordingly.

