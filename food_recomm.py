import time

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def app(data):
    st.title('Food Recommendation System')

    st.header('Enter items that you love very much and this recommender will suggest you different food items that you might like')

    st.image('images/food_image4.jpeg', use_column_width=True)

    fav_food = st.multiselect(
        label='Choose your favourite food items',
        options=data.loc[:, 'name'].values,
    )

    count = st.slider(
        label='Number of recipes to display',
        min_value=1,
        max_value=15,
        value=7,
        step=1
    )

    submit = st.button('Submit')

    if submit:
        if not fav_food:
            st.subheader('Please enter at least one food item')
        else:
            with st.spinner('Searching for recipes'):
                time.sleep(2)
                
                embeddings = []
                for q in fav_food:
                    food = data.loc[data['name'] == q].values[0][2:]
                    embeddings.append(food)

                embeddings = np.logical_or.reduce(embeddings)

                sim = cosine_similarity(data.iloc[:, 2:].values.reshape(255, -1), embeddings.reshape(1, -1)).ravel()

                idx_sorted = np.argsort(sim)[::-1]

                st.header('You might like these recipes')
                for val, idx in np.column_stack((sim[idx_sorted], idx_sorted)):
                    if count == 0:
                        break

                    if val > 0:
                        food = data.iloc[int(idx), 0]
                        if food not in fav_food:
                            st.info(f'**{data.iloc[int(idx), 0]}** ({data.iloc[int(idx), 1]})')
                            count -= 1

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)