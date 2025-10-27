# main.py
from modules.data_loader import DataLoader
from modules.analyzer import HousingAnalyzer
from modules.predictor import Predictor
from modules.visualizer import Visualizer
from modules.report_generator import ReportGenerator
from app_controller import AppController

if __name__ == "__main__":
    # Replace with your actual CSV path from data.gov.uk
    CSV_PATH = "data/UK-HPI-full-file-2025-04.csv"

    app = AppController(
        loader=DataLoader(CSV_PATH),
        analyzer=HousingAnalyzer(),
        predictor=Predictor(),
        visualizer=Visualizer(),
        reporter=ReportGenerator(),
    )

    # Example run – adjust region/property type to match your dataset values
    #app.run(region="London", property_type="Detached", forecast_months=6)

    app.run(region="London", property_type=None, forecast_months=6)
