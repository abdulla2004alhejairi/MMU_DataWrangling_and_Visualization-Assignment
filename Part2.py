
## Part 1 — Data Cleaning and Transformation 
import pandas as pd
import numpy as np

# Paths
AIRBNB_PATH = r"C:\Users\anson\Downloads\Assignment\AB_NYC_2019.csv"

# Load dataset
airbnb = pd.read_csv(AIRBNB_PATH)
print("Shape:", airbnb.shape)
airbnb.head()

#part Missing Data Handling
# Show missing values
airbnb_missing = airbnb.isnull().sum().sort_values(ascending=False)
display(airbnb_missing[airbnb_missing > 0])

# Apply chosen strategy
airbnb_cleaned = airbnb.dropna(subset=["name", "host_name"]).copy()
airbnb_cleaned["last_review"] = airbnb_cleaned["last_review"].fillna("No Review")
airbnb_cleaned["reviews_per_month"] = airbnb_cleaned["reviews_per_month"].fillna(0)

# Verify no missing left
airbnb_cleaned.isnull().sum()[airbnb_cleaned.isnull().sum() > 0]

# Part Outliers Detection and Treatment 

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
airbnb_cleaned["price"].plot(kind="box", vert=False)
plt.title("Boxplot of Price")

plt.subplot(1,2,2)
airbnb_cleaned["minimum_nights"].plot(kind="box", vert=False)
plt.title("Boxplot of Minimum Nights")
plt.tight_layout()
plt.show()

before = airbnb_cleaned.shape
airbnb_no_outliers = airbnb_cleaned[(airbnb_cleaned["price"] <= 1000) & (airbnb_cleaned["minimum_nights"] <= 365)]
after = airbnb_no_outliers.shape
print("Before:", before, "After:", after, "Removed:", before[0]-after[0])

#Part Data Transformation

# 1) & 2)
data_sub1 = airbnb_no_outliers[
    (airbnb_no_outliers["room_type"] == "Private room") &
    (airbnb_no_outliers["neighbourhood_group"] == "Manhattan")
]
data_sub1 = data_sub1[data_sub1["minimum_nights"] >= 3].sort_values(by="price")
display(data_sub1.head())

# 3) Average price per (neighbourhood_group, room_type)
data_sub2 = (airbnb_no_outliers
             .groupby(["neighbourhood_group", "room_type"], as_index=False)["price"]
             .mean()
             .rename(columns={"price": "avg_price"}))
display(data_sub2.head())

# 4) high_demand & revenue
airbnb_aug = airbnb_no_outliers.copy()
airbnb_aug["high_demand"] = airbnb_aug["availability_365"] > 300
airbnb_aug["revenue"] = airbnb_aug["price"] * airbnb_aug["minimum_nights"]
display(airbnb_aug[["neighbourhood_group", "room_type", "price", "minimum_nights", "availability_365", "high_demand", "revenue"]].head())

# 5) Bin price into categories
# Use quantiles for robust bins
bins = [-np.inf, airbnb_no_outliers["price"].quantile(0.33), airbnb_no_outliers["price"].quantile(0.66), np.inf]
labels = ["budget", "moderate", "expensive"]
airbnb_aug["price_bin"] = pd.cut(airbnb_aug["price"], bins=bins, labels=labels)
airbnb_aug["price_bin"].value_counts()

# 6) Summary stats of price by group
summary_price = (airbnb_no_outliers
                 .groupby(["neighbourhood_group", "room_type"])["price"]
                 .agg(["mean", "median"])
                 .reset_index())
display(summary_price.head())

# 7) Average availability_365 by neighbourhood_group
avg_availability = (airbnb_no_outliers
                    .groupby("neighbourhood_group")["availability_365"]
                    .mean()
                    .reset_index()
                    .rename(columns={"availability_365": "avg_availability_365"}))
display(avg_availability)
