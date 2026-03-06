import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Load data
df = pd.read_excel("final data.xlsx")  # Ensure your file is in the working directory

# Initialize geolocator with a proper user-agent
geolocator = Nominatim(user_agent="YourAppName_or_YourEmail@example.com")

# Function to get latitude and longitude
def get_lat_long(location):
    try:
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            return pd.Series([loc.latitude, loc.longitude])
    except (GeocoderTimedOut, GeocoderServiceError):
        return pd.Series([None, None])  # Return None if there's a timeout or service issue
    return pd.Series([None, None])

# Apply function to extract lat/lon
df[['Latitude', 'Longitude']] = df['LOCATION'].apply(lambda loc: get_lat_long(loc) if isinstance(loc, str) else pd.Series([None, None]))

# Save the updated DataFrame
df.to_excel("updated_data.xlsx", index=False)

print("Latitude and Longitude added successfully. Saved as 'updated_data.xlsx'")
