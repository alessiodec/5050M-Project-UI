# reading dataset

import pandas as pd

# The file ID from your Google Drive share link
file_id = "10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"

# Construct the direct download URL
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Use pandas to read the CSV file from the URL
df = pd.read_csv(url)

# Display the first few rows of the DataFrame
print(df.head())
