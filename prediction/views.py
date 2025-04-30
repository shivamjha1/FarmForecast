
from django.shortcuts import render, redirect
from .forms import CSVFileForm
from .models import CSVFile

# Create your views here.
# blog/views.py
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from django.conf import settings
import io
import json
import os
from django.conf import settings
from django.core.files.storage import default_storage
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def home(request):
    return render(request, 'prediction/template/page.html')

def homepage(request):
    return render(request, 'prediction/template/homepage.html')

def aboutus(request):
    return render(request, 'prediction/template/aboutus.html')

def upload_csv(request):
    if request.method == 'POST':
        form = CSVFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('visualize_csv')
    else:
        form = CSVFileForm()
    return render(request, 'prediction/template/upload.html', {'form': form})




def visualize_csv(request):
    # Assuming only one file for simplicity
    latest_file = CSVFile.objects.latest('uploaded_at')
    csv_path = latest_file.file.path

    # Read CSV into DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Group data by State and Commodity
    grouped_data = df.groupby(['State', 'Commodity'])

    # Prepare data for visualization
    graphs_data = []
    for (state, commodity), group in grouped_data:
        group_sorted = group.sort_values('Date')  # Sort by date
        graphs_data.append({
            'state': state,
            'commodity': commodity,
            'dates': group_sorted['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': group_sorted['Predicted_Modal_Price'].tolist(),
        })

    return render(request, 'prediction/template/visualize_csv.html', {
        'graphs_data': graphs_data,
    })





# Function to load label encoder
def load_label_encoder(csv_file_path):
    df = pd.read_csv(csv_file_path)
    le = LabelEncoder()
    le.classes_ = df['original_value'].values
    return le

# Function to handle file upload, prediction, and CSV download
def file_upload_view(request, category):
    if request.method == 'POST' and request.FILES.get('file-upload'):
        uploaded_file = request.FILES['file-upload']
        try:
            # Save the file temporarily
            file_path = default_storage.save('temp/'+uploaded_file.name, uploaded_file)

            # Load the uploaded CSV file
            df_cleaned = pd.read_csv(file_path)

            # Load your pre-trained model
            model = load_model(os.path.join(settings.BASE_DIR, f'{category}_model.h5'), compile=False)  # Adjust model path

            # Load label encoders for State and Commodity from CSV
            le_state = load_label_encoder(os.path.join(settings.BASE_DIR, f'{category}_state_encoded.csv'))
            le_commodity = load_label_encoder(os.path.join(settings.BASE_DIR, f'{category}_commodity_encoded.csv'))

            # Fit the scaler on the price columns
            scaler = MinMaxScaler()
            scaler.fit(df_cleaned[['Min_Price', 'Max_Price', 'Modal_Price']])

            # Convert 'Arrival_Date' to datetime
            df_cleaned['Arrival_Date'] = pd.to_datetime(df_cleaned['Arrival_Date'], format='%d-%m-%Y')

            # Call the prediction function to get predicted prices
            predictions_df = predict_prices_for_all_states_and_commodities(
                model,
                file_path,
                le_state,
                le_commodity,
                scaler,
                n_days=10
            )

            # Handle download request
            if 'download' in request.GET and request.GET['download'] == 'csv':
                # Create a CSV response for download
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="predicted_prices.csv"'
                predictions_df.to_csv(path_or_buf=response, index=False)
                return response

            # Prepare data for charting
            predictions_json = predictions_df.to_json(orient='records')  # Convert DataFrame to JSON format for charts

            # Convert predictions to HTML
            predictions_html = predictions_df.to_html(classes="table table-bordered", index=False)

            return render(request, 'prediction/template/result_page.html', {
                'predictions': predictions_html,  # Pass the JSON data for the charts
            })

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")

    return render(request, 'prediction/template/upload_page.html')


# Define the prediction function to handle the prediction logic
def predict_prices_for_all_states_and_commodities(model, file_path, le_state, le_commodity, scaler, n_days=10):
    # Load the data
    new_data = pd.read_csv(file_path)

    # Convert 'Arrival_Date' to datetime and sort by date
    new_data['Arrival_Date'] = pd.to_datetime(new_data['Arrival_Date'], format='%d-%m-%Y')
    new_data = new_data.sort_values(by='Arrival_Date')

    # Encode categorical variables ('State' and 'Commodity')
    new_data['State_encoded'] = le_state.transform(new_data['State'])
    new_data['Commodity_encoded'] = le_commodity.transform(new_data['Commodity'])

    # Normalize price columns using the scaler
    new_data[['Min_Price', 'Max_Price', 'Modal_Price']] = scaler.transform(
        new_data[['Min_Price', 'Max_Price', 'Modal_Price']])
    
    # Prepare a list to store the results
    results = []

    # Group data by 'State' and 'Commodity'
    grouped_data = new_data.groupby(['State', 'Commodity'])

    # Iterate through each state-commodity group
    for (state, commodity), group in grouped_data:
        if len(group) < 30:
            continue  # Skip if insufficient data

        last_30_days_data = group.tail(30)[
            ['Min_Price', 'Max_Price', 'Modal_Price', 'State_encoded', 'Commodity_encoded']
        ].values

        if last_30_days_data.shape != (30, 5):
            continue  # Skip if the shape is incorrect

        state_encoded = le_state.transform([state])[0]
        commodity_encoded = le_commodity.transform([commodity])[0]

        predicted_prices = predict_next_n_days(
            model, last_30_days_data, state_encoded, commodity_encoded, n_days=n_days
        )

        last_date = group['Arrival_Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days).tolist()

        for date, price in zip(future_dates, predicted_prices):
            results.append([state, commodity, date, price])

    predictions_df = pd.DataFrame(results, columns=['State', 'Commodity', 'Date', 'Predicted_Modal_Price'])

    predictions_df['Predicted_Modal_Price'] = scaler.inverse_transform(
        np.concatenate([
            np.zeros((len(predictions_df), 2)),
            predictions_df[['Predicted_Modal_Price']].values
        ], axis=1)
    )[:, 2]

    predictions_df['Predicted_Modal_Price'] = predictions_df['Predicted_Modal_Price'].apply(lambda x: f"{x:.2f}")
    
    return predictions_df


# Define the function for predicting the next n days
def predict_next_n_days(model, last_n_days_data, state_encoded, commodity_encoded, n_days=10):
    predictions = []
    input_data = np.copy(last_n_days_data)

    for _ in range(n_days):
        input_data_reshaped = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))
        predicted_price = model.predict(input_data_reshaped)
        predictions.append(predicted_price[0][0])
        next_input = np.array([0, 0, predicted_price[0][0], state_encoded, commodity_encoded])
        input_data = np.vstack((input_data[1:], next_input))

    return predictions



