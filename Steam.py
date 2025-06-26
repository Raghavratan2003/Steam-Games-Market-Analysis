import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r'C:\Users\Raghav Ratan Yadav\Downloads\archive (1)\steam.csv')

# ===================
# Data Cleaning
# ===================
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])
df['release_year'] = df['release_date'].dt.year

df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
df['price_category'] = df['price'].apply(lambda x: 'Free' if x == 0 else 'Paid')

df['genres'] = df['genres'].fillna('Unknown')
df['genres_split'] = df['genres'].str.split(';')

# ===================
# EDA Visualizations
# ===================

# 1️⃣ Price distribution
plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=range(0, int(df['price'].max())+5, 5), color='teal')
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Number of Games')
plt.show()


# 2️⃣ Games released per year
df.groupby('release_year').size().plot(marker='o')
plt.title('Games Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show()

# 3️⃣ Free vs Paid
df['price_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['gold', 'lightblue'])
plt.title('Free vs Paid Games')
plt.ylabel('')
plt.show()

# 4️⃣ Top genres
exploded = df.explode('genres_split')
exploded['genres_split'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Games')
plt.show()

# 5️⃣ Avg price by genre
exploded.groupby('genres_split')['price'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='orchid')
plt.title('Top 10 Genres by Avg Price')
plt.ylabel('Avg Price ($)')
plt.show()

# 6️⃣ Price vs positive ratings
plt.scatter(df['price'], df['positive_ratings'], alpha=0.5, color='purple')
plt.title('Price vs Positive Ratings')
plt.xlabel('Price ($)')
plt.ylabel('Positive Ratings')
plt.show()

# 7️⃣ Correlation heatmap
corr = df[['price', 'positive_ratings', 'negative_ratings']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ===================
# Advanced Insights
# ===================

# Undervalued genres (low price + high rating)
avg_price = exploded.groupby('genres_split')['price'].mean()
avg_rating = exploded.groupby('genres_split')['positive_ratings'].mean()
undervalued = pd.concat([avg_price, avg_rating], axis=1).dropna()
undervalued.columns = ['avg_price', 'avg_rating']
print(undervalued.sort_values(by='avg_rating', ascending=False).head())

# Price trend over years
df.groupby('release_year')['price'].mean().plot(marker='o', color='darkgreen')
plt.title('Average Price Over Years')
plt.ylabel('Avg Price ($)')
plt.show()

# Free vs paid stacked bar over time
stacked = df.groupby(['release_year', 'price_category']).size().unstack(fill_value=0)
stacked.plot(kind='bar', stacked=True, figsize=(10,6), color=['gold', 'lightblue'])
plt.title('Free vs Paid Games Released Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show()