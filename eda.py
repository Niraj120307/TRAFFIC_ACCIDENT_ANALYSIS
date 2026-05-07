import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import plotly.express as px
import pandas as pd
df = pd.read_csv('data/clean_accidents.csv')

# Hotspot heatmap
m = folium.Map(location=[37.0, -95.0], zoom_start=4)
sample = df[['Start_Lat', 'Start_Lng']].dropna().sample(50000)
HeatMap(sample.values.tolist(), radius=8).add_to(m)
m.save('outputs/hotspot_map.html')

# Hour of day chart
plt.figure(figsize=(12, 4))
df['hour'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Accidents by hour of day')
plt.tight_layout()
plt.savefig('outputs/hourly_trend.png')

# Hour x Day heatmap
pivot = df.groupby(['hour', 'day_of_week']).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3)
plt.title('Accidents: hour vs day of week')
plt.tight_layout()
plt.savefig('outputs/hour_day_heatmap.png')

# Weather vs severity
top_weather = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(15)
top_weather.plot(kind='barh', figsize=(8, 5), color='coral')
plt.title('Avg severity by weather condition')
plt.tight_layout()
plt.savefig('outputs/weather_severity.png')