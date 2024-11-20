import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
listings = pd.read_csv("listings.csv")

# Clean and convert the 'price' column
def clean_price(price):
    try:
        return float(price.replace("$", "").replace(",", "").strip())
    except:
        return np.nan

listings['price'] = listings['price'].apply(clean_price)
listings = listings.dropna(subset=['price'])

# Group by Neighborhood and Calculate Average Price
neighborhood_prices = listings.groupby('neighbourhood_cleansed').agg(
    avg_price=('price', 'mean')
).reset_index()

# Sort by Average Price
top_neighborhoods = neighborhood_prices.nlargest(20, 'avg_price')

# Bar Graph: Average Price by Neighborhood
plt.figure(figsize=(10, 6))
sns.barplot(data=top_neighborhoods, x='neighbourhood_cleansed', y='avg_price', palette='viridis')
plt.title("Average Price by Neighborhood in Chicago")
plt.xlabel("Neighborhood")
plt.ylabel("Average Price (USD)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Group By price category
def categorize_price(price):
    if price < 100:
        return 'Budget'
    elif 100 <= price < 300:
        return 'Mid-range'
    else:
        return 'Luxury'

listings['price_category'] = listings['price'].apply(categorize_price)


price_category_summary = listings.groupby('price_category').agg(
    avg_price=('price', 'mean'),
    total_listings=('id', 'count')
).reset_index()

# Bar Graph for Price Categories
plt.figure(figsize=(8, 6))
sns.barplot(data=price_category_summary, x='price_category', y='avg_price', palette='coolwarm')
plt.title("Average Price by Price Category")
plt.xlabel("Price Category")
plt.ylabel("Average Price (USD)")
plt.tight_layout()
plt.show()


listings['review_scores_rating'] = np.ceil(listings['review_scores_rating'])

average_price_by_score = listings.groupby('review_scores_rating').agg(
    avg_price=('price', 'mean')
).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=average_price_by_score, x='review_scores_rating', y='avg_price', palette='viridis')
plt.title("Average Price by Review Score")
plt.xlabel("Review Score (Rounded)")
plt.ylabel("Average Price (USD)")
plt.tight_layout()
plt.show()


# Create bins for ranges of availability
availability_bins = pd.cut(
    listings['availability_365'],
    bins=[0, 90, 180, 270, 365],
    labels=['0-90 days', '91-180 days', '181-270 days', '271-365 days']
)

# Distribution of Listings by Availability
availability_distribution = availability_bins.value_counts().sort_index()

# Plot a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=availability_distribution.index, y=availability_distribution.values, palette='coolwarm')
plt.title("Distribution of Listings by Availability")
plt.xlabel("Availability Range (Days)")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.show()



#Price Distribution
#The spread of listing prices to identify common price ranges and outliers.
plt.figure(figsize=(10, 6))
sns.histplot(listings['price'], bins=50, kde=True, color='blue')
plt.title("Distribution of Listing Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


#Count of Listings by Room Type
room_type_counts = listings['room_type'].value_counts().reset_index()
room_type_counts.columns = ['room_type', 'count']
plt.figure(figsize=(8, 6))
sns.barplot(data=room_type_counts, x='room_type', y='count', palette='coolwarm')
plt.title("Count of Listings by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.show()

#price vs review score Whether higher-rated listings tend to have higher prices.


plt.figure(figsize=(10, 6))
sns.scatterplot(data=listings, x='review_scores_rating', y='price', alpha=0.6, color='green')
plt.title("Price vs. Review Scores")
plt.xlabel("Review Scores")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.show()



