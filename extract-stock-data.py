import yfinance as yf  # Import the yfinance package
import csv  # Import the csv module
import datetime  # Import the datetime module

# Create a Ticker object for the stock you want to get data for
ticker = "TSLA"  # Ticker symbol for Apple
stock = yf.Ticker(ticker)

# Use the Ticker object's history method to get the stock data
# Specify the time period and interval for the data
# For example, to get the stock data for the past month in daily intervals:
period = "10y"  # Past month
interval = "1d"  # Daily intervals
stock_data = stock.history(period=period, interval=interval)

# Store the original column names
columns = stock_data.columns.tolist()

# Modify the index
modified_index = []
for date in stock_data.index:
    # Convert the date object to a string
    date = str(date)

    # Convert the date string to a datetime object using the correct format string
    date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")

    # Format the date as a string in the desired format
    formatted_date = date.strftime("%Y/%m/%d")

    # Add the formatted date to the list
    modified_index.append(formatted_date)

# Set the index of the dataframe using the modified index
stock_data.index = modified_index

# Add the modified index to the list of columns
columns = ["Date"] + columns

# Create a file name for the CSV file based on the ticker symbol and time period
file_name = f"{ticker}_{period}_{interval}.csv"

# Open the CSV file in write mode
with open(file_name, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the column names to the CSV file
    writer.writerow(columns)

    # Iterate through the rows of the stock data
    for date, row in stock_data.iterrows():
        # Write the date and row data to the CSV file
        writer.writerow([date] + row.tolist())
