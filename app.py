import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import food_recomm
import recipe_finder

st.set_page_config(
    page_title="FFF",
    page_icon=':pizza:'
)

@st.cache_data()
def fetch_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # get all items
    items = set()
    for x in df.ingredients:
        for val in x.split(', '):
            items.add(val.lower().strip())
        # break
    # items = sorted(items)

    # create new dataframe
    new_df = pd.DataFrame(data=np.zeros((255, 367), dtype=int), columns=['name', 'ingredients'] + list(items))


    for i, d in df.iterrows():
        new_df.loc[i, ['name', 'ingredients']] = d[:2]

        for val in d[1].split(', '):
            item = val.lower().strip()
            new_df.loc[i, item] = 1

    return new_df


data = fetch_and_clean_data('data/food_250.csv')

PAGES = {
    'Recipe Finder': recipe_finder,
    'Food Recommender': food_recomm
}

page = st.sidebar.radio(
    label='Contents',
    options=list(PAGES.keys())
)



content = PAGES[page]
content.app(data)
