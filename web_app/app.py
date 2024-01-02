import streamlit as st
import torch
from PIL import Image

from web_app_utils import create_encoding_dict, classify_image


st.set_page_config(layout="centered")


# ______________________________ Head of page ______________________________
st.title('Architectural Style Classification')
description = """
This architectural style classification model was developed as part of the course Applied Deep Learning at TU Wien. The 
approach combines the MobileNets architecture with a channel-spatial attention mechanism. The model achieves an accuracy 
of 58%.

### Please upload an image of a building.
"""
st.write(description)


# ______________________________ Classification ______________________________
# Load model
model = torch.jit.load('web_app/best_found_architectural_style_classification_model.pt')
model.eval()

# Load class encodings
encoding_dict = create_encoding_dict('web_app/class_encoding.csv')

# File upload
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load and display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width="auto")

    # Classify
    predicted_class, probability = classify_image(model, encoding_dict, image)

    # Write classification
    st.write("### {}".format(predicted_class))
    st.write("#### Score: {}%".format(int(probability * 1000) / 10))

# ______________________________ Infos and Instructions ______________________________
info = """
Information on model use and performance...

Classes:
* XXX
* XXX

"""
st.info(info, icon="ℹ️")
