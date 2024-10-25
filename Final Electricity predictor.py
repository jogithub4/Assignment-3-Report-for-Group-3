# Online 3 u3222271, u3242961, u3143854, u3204559
# Assessment 3: Part A - Report, Electricity Predictor app
# 25/10/2024
# Software Technology 1

import tkinter as tk
from tkinter import ttk
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('final_rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

class ElectricityPrediction:
    def __init__(self, master):
        self.master = master
        master.title("Electricity Price Prediction App")

        self.style = ttk.Style()
        self.style.configure("TLabel", padding=5, font=('Arial', 10))
        self.style.configure("TEntry", padding=5, font=('Arial', 10))
        self.style.configure("TButton", padding=10, font=('Arial', 10))
        self.style.configure("TFrame", padding=10)

        self.holiday_var = tk.StringVar(value="0")
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.widgets()

    def widgets(self):
        ttk.Label(self.main_frame, text="Electricity Price Prediction", font=('Arial', 15, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        holiday_frame = ttk.LabelFrame(self.main_frame, text="Holiday?")
        holiday_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=10)
        ttk.Radiobutton(holiday_frame, text="Yes", variable=self.holiday_var, value="1").grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(holiday_frame, text="No", variable=self.holiday_var, value="0").grid(row=0, column=1, padx=5, pady=5)

        input_frame = ttk.LabelFrame(self.main_frame, text="Input Values")
        input_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Label(input_frame, text="Day of Week (1-7):").grid(row=0, column=0, sticky=tk.W)
        self.day_of_week_entry = ttk.Entry(input_frame)
        self.day_of_week_entry.grid(row=0, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="Week of Year (1-52):").grid(row=1, column=0, sticky=tk.W)
        self.week_of_year_entry = ttk.Entry(input_frame)
        self.week_of_year_entry.grid(row=1, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="Forecast Wind Production:").grid(row=2, column=0, sticky=tk.W)
        self.forecast_wind_production_entry = ttk.Entry(input_frame)
        self.forecast_wind_production_entry.grid(row=2, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="System Load EA:").grid(row=3, column=0, sticky=tk.W)
        self.system_load_ea_entry = ttk.Entry(input_frame)
        self.system_load_ea_entry.grid(row=3, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="SMPEA:").grid(row=4, column=0, sticky=tk.W)
        self.smpea_entry = ttk.Entry(input_frame)
        self.smpea_entry.grid(row=4, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="ORK Temperature:").grid(row=5, column=0, sticky=tk.W)
        self.ork_temperature_entry = ttk.Entry(input_frame)
        self.ork_temperature_entry.grid(row=5, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="ORK Windspeed:").grid(row=6, column=0, sticky=tk.W)
        self.ork_windspeed_entry = ttk.Entry(input_frame)
        self.ork_windspeed_entry.grid(row=6, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="CO2 Intensity:").grid(row=7, column=0, sticky=tk.W)
        self.co2_intensity_entry = ttk.Entry(input_frame)
        self.co2_intensity_entry.grid(row=7, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="Actual Wind Production:").grid(row=8, column=0, sticky=tk.W)
        self.actual_wind_production_entry = ttk.Entry(input_frame)
        self.actual_wind_production_entry.grid(row=8, column=1, sticky=tk.E)

        ttk.Label(input_frame, text="System Load EP2:").grid(row=9, column=0, sticky=tk.W)
        self.system_load_ep2_entry = ttk.Entry(input_frame)
        self.system_load_ep2_entry.grid(row=9, column=1, sticky=tk.E)

        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        predict_button = ttk.Button(button_frame, text="Predict Price", command=self.predict)
        predict_button.pack(side=tk.LEFT, padx=5)
        quit_button = ttk.Button(button_frame, text="Quit", command=self.master.destroy)
        quit_button.pack(side=tk.LEFT, padx=5)

        self.result_label = ttk.Label(self.main_frame, text="", font=('Arial', 14))
        self.result_label.grid(row=4, column=0, columnspan=2)

    def predict(self):
        try:
            try:
                holiday_flag = float(self.holiday_var.get())
                day_of_week = float(self.day_of_week_entry.get())
                week_of_year = float(self.week_of_year_entry.get())
                forecast_wind_production = float(self.forecast_wind_production_entry.get())
                system_load_ea = float(self.system_load_ea_entry.get())
                smpea = float(self.smpea_entry.get())
                ork_temperature = float(self.ork_temperature_entry.get())
                ork_windspeed = float(self.ork_windspeed_entry.get())
                co2_intensity = float(self.co2_intensity_entry.get())
                actual_wind_production = float(self.actual_wind_production_entry.get())
                system_load_ep2 = float(self.system_load_ep2_entry.get())
            except ValueError:
                raise ValueError("Error, please make sure inputs are positive numbers.")

            try:
                if not 1 <= day_of_week <= 7:
                    raise ValueError("Error: Day of Week must be between 1 and 7.")
                if not 1 <= week_of_year <= 52:
                    raise ValueError("Error: Week of Year must be between 1 and 52.")

                input_data = pd.DataFrame([[holiday_flag, day_of_week, week_of_year, forecast_wind_production,
                                            system_load_ea, smpea, ork_temperature, ork_windspeed,
                                            co2_intensity, actual_wind_production, system_load_ep2]],
                                            columns=['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'ForecastWindProduction',
                                                    'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed',
                                                    'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2'])

                scaled_data = scaler.transform(input_data)
                prediction_result = model.predict(scaled_data)[0]
                self.result_label.config(text=f"The predicted price for this day is: ${prediction_result:.2f}")

            except ValueError:
                if not 1 <= day_of_week <= 7:
                    self.result_label.config(text="Error: Day of Week must be between 1 and 7.")
                elif not 1 <= week_of_year <= 52:
                    self.result_label.config(text="Error: Week of Year must be between 1 and 52.")
                return

        except ValueError:
            self.result_label.config(text=f"Error, please make sure inputs are positive numbers.")

root = tk.Tk()
app = ElectricityPrediction(root)
root.mainloop()
