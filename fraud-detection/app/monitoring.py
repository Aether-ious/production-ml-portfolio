from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import pandas as pd
import joblib
from pathlib import Path

REFERENCE_DATA_PATH = Path(__file__).parent.parent / "data" / "reference_sample.pkl"

reference_data = joblib.load(REFERENCE_DATA_PATH)

def generate_drift_report(current_data: pd.DataFrame) -> str:
    report = Report(metrics=[
        DataDriftTable(),
        DatasetDriftMetric()
    ])
    report.run(reference_data=reference_data, current_data=current_data)
    report_path = Path(__file__).parent.parent / "evidently_reports" / "drift_report.html"
    report.save_html(report_path)
    return str(report_path)