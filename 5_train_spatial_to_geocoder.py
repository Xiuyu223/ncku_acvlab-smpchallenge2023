import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="geoapiExercises")

df_old = pd.read_csv('train/category_none_new_old_train_processed_na_time.csv', encoding='utf-8-sig')
df_reorder=df_old.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1','Unnamed: 0'])
print(df_reorder.head())
print(len(np.unique(df_reorder['Longitude'].values)))


def geocode_location(row):

    try:

        if row['Latitude'] == 0 or row['Longitude'] == 0:

            return pd.Series({'City': 'NONE', 'State': 'NONE', 'Country': 'NONE'})
        else:
            location = geolocator.reverse(f"{row['Latitude']},{row['Longitude']}")
            address = location.raw['address']
            city = address.get('city', 'NONE')
            state = address.get('state', 'NONE')
            country = address.get('country', 'NONE')

            print(country,state,city)
            return pd.Series({'City': city, 'State': state, 'Country': country})
    except:
        print('error')
        return pd.Series({'City': 'NONE', 'State': 'NONE', 'Country': 'NONE'})


df_reorder[['Longitude', 'Latitude']] = df_reorder[['Longitude', 'Latitude']].fillna(0)

print(len(np.unique(df_reorder['Longitude'])))
df_reorder[['City', 'State', 'Country']] = df_reorder.apply(geocode_location, axis=1)

df_reorder=df_reorder.drop(columns=['Longitude', 'Latitude'])
df_reorder.to_csv('train/geo_train_processed_na_time.csv',
                  index=False)
