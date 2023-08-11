import time

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from helper import embed_query


def app(data):
    st.title('Food From Fridge:fire:')

    st.image('images/items_image.jpg', use_column_width=True)

    available_items = st.multiselect(
        label = 'Select items that are available with you',
        options = data.columns[2:],
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
        if not available_items:
            st.subheader('Please enter atleast 2 items')
        else:
            with st.spinner('Searching for recipes'):
                time.sleep(2)
                emb_qy = embed_query(available_items, data.columns[2:].values)
                sim = cosine_similarity(data.iloc[:, 2:].values.reshape(255, -1), emb_qy.reshape(1, -1)).ravel()

                idx_sorted = np.argsort(sim)[::-1]

                st.header('You can try these recipes:sunglasses:')
                for val, idx in np.column_stack((sim[idx_sorted], idx_sorted)):
                    if count and 0 < val < 0.99:
                        st.info(f'**{data.iloc[int(idx), 0]}** ({data.iloc[int(idx), 1]})')
                        count -= 1



                        #You can try to make these recipes
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)