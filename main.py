import streamlit as st
from keras.models import load_model
from PIL import ImageOps, Image
import numpy as np


# Caractericas Basicas de la Pagina
st.set_page_config(
    page_icon="🤖", page_title="App Afecciones Cerebrales", layout="wide")


with st.container():
    st.image("img/Logo-UPB-2022.png", width=250)
    st.title("Pronóstico De Afecciones Cerebrales🤖")
    st.subheader("Programa De Ingeniería Electrónica")
    st.write(f"""Presentado Por: Juan Diego Orozco Gomez""")

# Path del modelo preentrenado
MODEL_PATH = 'models/clasificacion_de_afecciones_cerebrales3.h5'

# Tamaños de las Imagenes
width_shape = 150
height_shape = 150

# cargar clasificador
model = load_model(MODEL_PATH)

# nombres de la clasificacion
names = ["El cerebro del paciente se encuentra con ISQUEMIA CEREBRAL",
         "El estado del cerebro del paciente es NORMAL",
         "El cerebro del paciente se encuentra con ENCEFALOPATÍA POSTERIOR REVERSIBLE (PRES)",
         "El cerebro del paciente se encuentra con NEUROMIELITIS OPTICA"
         ]

def model_prediction(img, model):

    # Normalizar imagenes
    img_resize = ImageOps.fit(
        img, (width_shape, height_shape), Image.Resampling.LANCZOS)
    img_array = np.array(img_resize)
    img_array = img_array.reshape(1, width_shape, height_shape, 3)
    # hacer predicción
    preds = model.predict(img_array)
    return preds


def main():

    model = ''

    if model == '':
        model = load_model(MODEL_PATH)

    predictS = ""
    # subir archivo
    file = st.file_uploader('Carga una imagen', type=['jpeg', 'jpg', 'png'])

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Imagen", use_column_width=False)

    if st.button("Predicción"):
        predictS = model_prediction(image, model)
        st.success('### {}'.format(names[np.argmax(predictS)]))

    st.markdown('<a href="https://portafolio-osweb.netlify.app/" style="text-decoration: none;"><p class="footer__copy"> © Orlando Ospino H - 2023</p></a>',unsafe_allow_html=True)


if __name__ == '__main__':
    main()
