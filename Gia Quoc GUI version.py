import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

class DiamondsPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Diamonds Price Prediction')

        file_path = 'diamonds.csv'
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File '{file_path}' not found.")
            return

        self.data = pd.read_csv(file_path)
        
        # Drop the unnamed index column if it exists
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.drop(columns=['Unnamed: 0'])

        self.features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
        self.inputs = {}

        # Convert categorical features to numeric using get_dummies
        self.X = pd.get_dummies(self.data.drop('price', axis=1), drop_first=True)
        self.y = self.data['price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, feature in enumerate(self.features):
            label = tk.Label(self.master, text=f'{feature.capitalize()}:')
            label.grid(row=i, column=0, sticky=tk.W, pady=2)

            if feature in ['carat', 'depth', 'table', 'x', 'y', 'z']:
                entry = tk.Entry(self.master)
                entry.grid(row=i, column=1, pady=2)
                self.inputs[feature] = entry
            elif feature == 'cut':
                cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
                cut_combobox = ttk.Combobox(self.master, values=cut_options)
                cut_combobox.grid(row=i, column=1, pady=2)
                self.inputs[feature] = cut_combobox
            elif feature == 'color':
                color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
                color_combobox = ttk.Combobox(self.master, values=color_options)
                color_combobox.grid(row=i, column=1, pady=2)
                self.inputs[feature] = color_combobox
            elif feature == 'clarity':
                clarity_options = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2']
                clarity_combobox = ttk.Combobox(self.master, values=clarity_options)
                clarity_combobox.grid(row=i, column=1, pady=2)
                self.inputs[feature] = clarity_combobox

        predict_button = tk.Button(self.master, text='Submit', command=self.predict_price)
        predict_button.grid(row=len(self.features), column=1, pady=10)

    def predict_price(self):
        # Prepare inputs for prediction
        input_data = {}
        for feature in self.features:
            value = self.inputs[feature].get()
            if feature in ['carat', 'depth', 'table', 'x', 'y', 'z']:
                input_data[feature] = [float(value)]
            else:
                for category in self.X.columns:
                    if category.startswith(feature + '_'):
                        input_data[category] = [1 if category.endswith(value) else 0]

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Ensure all columns are present
        for column in self.X.columns:
            if column not in input_df.columns:
                input_df[column] = 0

        # Order columns as in training data
        input_df = input_df[self.X.columns]

        # Predict price
        try:
            price = self.model.predict(input_df)
            messagebox.showinfo('Predicted Price', f'The predicted diamond price is ${price[0]:.2f}')
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = DiamondsPricePredictionApp(root)
    root.mainloop()
