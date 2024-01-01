import streamlit as st

# set title
st.title('Architectural Style Classification')

# set header
st.header('Please upload an image of a building.')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
