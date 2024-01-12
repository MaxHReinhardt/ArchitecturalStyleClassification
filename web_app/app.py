import streamlit as st
import torch
from PIL import Image

from web_app_utils import create_encoding_dict, create_info_dict, classify_image


st.set_page_config(layout="centered")


# ______________________________ Head of page ______________________________
st.title('Architectural Style Classification')
description = """
This architectural style classification model was developed as part of the course Applied Deep Learning at TU Wien. The 
development aimed at combining low computational requirements with a reasonable accuracy. 

To test the model, a picture of a building can be uploaded below; the application will show the predicted architectural 
style, a percentage score reflecting the confidence of the model and a small info text. For best results, upload an 
image that shows all parts of the building and contains as little background or additional objects as possible.
A list with all styles that can be predicted can be found below.

Please be aware that the model can make mistakes! Its accuracy is 58%.

More information/code on GitHub: https://github.com/MaxHReinhardt/ArchitecturalStyleClassification

### Please upload an image of a building.
"""
st.write(description)


# ______________________________ Classification ______________________________
# Load model
model = torch.jit.load('web_app/best_found_architectural_style_classification_model.pt')
model.eval()

# Load class encodings
encoding_dict = create_encoding_dict('web_app/class_encoding.csv', delimiter=',')
info_dict = create_info_dict('web_app/style_info.csv', delimiter=';')

# File upload
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load and display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width="auto")

    # Classify
    predicted_class, probability = classify_image(model, encoding_dict, image)
    info, link = info_dict[predicted_class][0], info_dict[predicted_class][1]

    # Write classification
    st.write("### {}".format(predicted_class))
    st.write("#### Score: {}%".format(int(probability * 1000) / 10))
    st.write(f"**Info:** {info} \n\n **Retrieved from:** {link}")


# ______________________________ Infos and Instructions ______________________________
st.divider()
class_info_text = """
**The model can predict the following styles**:
* Palladian architecture
* Novelty architecture
* International style
* Chicago school architecture
* Beaux-Arts architecture
* Ancient Egyptian architecture
* Gothic architecture
* Tudor Revival architecture
* Romanesque architecture
* Colonial architecture
* American craftsman style
* Greek Revival architecture
* Queen Anne architecture
* Baroque architecture
* Edwardian architecture
* Art Nouveau architecture
* Deconstructivism
* Bauhaus architecture
* Georgian architecture
* Byzantine architecture
* Postmodern architecture
* Achaemenid architecture
* Art Deco architecture
* American Foursquare architecture
* Russian Revival architecture
"""

st.write(class_info_text)

