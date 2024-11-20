import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
listings = pd.read_csv("listings.csv")

# Define Chicago as the tourist landmark
chicago_landmark = (41.8781, -87.6298)  # Chicago Latitude and Longitude

### 1. Feature Enrichment ###
# Amenity Scoring
def calculate_amenity_score(amenities):
    if isinstance(amenities, str):
        amenities_list = amenities.strip("{}").split(",")
        return len(amenities_list)  # Count number of amenities
    return 0

listings['amenity_score'] = listings['amenities'].apply(calculate_amenity_score)

# Proximity to Chicago Landmark
def calculate_proximity(lat, lon, landmark):
    try:
        return geodesic((lat, lon), landmark).km
    except:
        return np.nan

listings['distance_to_chicago'] = listings.apply(
    lambda row: calculate_proximity(row['latitude'], row['longitude'], chicago_landmark), axis=1
)

def clean_price(price):
    try:
        return float(price.replace("$", "").replace(",", "").strip())
    except:
        return np.nan

listings['price'] = listings['price'].apply(clean_price)

# Drop rows with NaN in 'price'
listings = listings.dropna(subset=['price'])

# Categorize Listings by Price Range
def categorize_price(price):
    if price < 100:
        return 'Budget'
    elif 100 <= price < 300:
        return 'Mid-range'
    else:
        return 'Luxury'

listings['price_category'] = listings['price'].apply(categorize_price)


### 3. Market Segmentation ###
# Aggregate Metrics by Neighborhood
neighborhood_analysis = listings.groupby('neighbourhood').agg(
    avg_price=('price', 'mean'),
    avg_review_score=('review_scores_rating', 'mean'),
    total_listings=('id', 'count')
).reset_index()

### 4. Occupancy Rate Estimation ###
# Assuming review_scores_rating proxies for popularity
listings['occupancy_rate_estimate'] = (
    listings['review_scores_rating'] / 100 * listings['number_of_reviews']
)

### 5. Visualization ###
# Price Distribution by Category
sns.boxplot(data=listings, x='price_category', y='price')
plt.title("Price Distribution by Category")
plt.show()

# Proximity vs. Price
sns.scatterplot(data=listings, x='distance_to_chicago', y='price', hue='price_category')
plt.title("Price vs. Proximity to Chicago")
plt.show()

# Neighborhood Analysis
neighborhood_analysis.sort_values('avg_price', ascending=False, inplace=True)
sns.barplot(data=neighborhood_analysis, x='neighbourhood', y='avg_price')
plt.xticks(rotation=90)
plt.title("Average Price by Neighborhood")
plt.show()

# Save Results
listings.to_csv("transformed_listings_chicago.csv", index=False)
neighborhood_analysis.to_csv("neighborhood_analysis_chicago.csv", index=False)
