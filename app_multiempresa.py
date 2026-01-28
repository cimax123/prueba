import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import os
import io

def check_password():
    try:
        correct_password = st.secrets["password"]
    except (FileNotFoundError, KeyError):
        correct_password = "test"
        st.warning("Advertencia: No se encontraron los 'secrets'. Usando contraseña de prueba local.")

    st.title('🤖 Asistente Contable Multi-Empresa')
    password = st.text_input("Ingresa la contraseña para acceder:", type="password")

    if not password:
        st.stop()

    if password == correct_password:
        return True
    else:
        if password:
            st.error("La contraseña es incorrecta.")
        st.stop()

@st.cache_resource
def cargar_y_entrenar_modelo(ruta_archivo_empresa):
    try:
        df_train = pd.read_excel(ruta_archivo_empresa, header=None, names=['numero_cuenta', 'descripcion', 'nombre_cuenta'])
        df_train.dropna(subset=['descripcion', 'nombre_cuenta'], inplace=True)
        df_train['descripcion'] = df_train['descripcion'].astype(str).str.lower()
        counts = df_train['nombre_cuenta'].value_counts()
        df_filtrado = df_train[df_train['nombre_cuenta'].isin(counts[counts >= 2].index)]

        if df_filtrado.empty:
            st.error(f"El archivo de entrenamiento para esta empresa no tiene suficientes datos para entrenar.")
            return None

        X = df_filtrado['descripcion']
        y = df_filtrado['nombre_cuenta']
        model = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ('classifier', LinearSVC(random_state=42, class_weight='balanced'))
        ])
        model.fit(X, y)
        print(f"Modelo entrenado para {ruta_archivo_empresa}")
        return model
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo de entrenamiento: {e}")
        return None

def clasificar_datos(modelo, df, nombre_columna):
    """Clasifica los datos del dataframe usando el modelo proporcionado."""
    try:
        descripciones = df[nombre_columna].astype(str).str.lower()
        predicciones = modelo.predict(descripciones)
        df['cuenta_sugerida'] = predicciones
        return df
    except Exception as e:
        raise e

def main():
    st.set_page_config(
        page_title="Asistente Contable Multi-Empresa",
        layout="wide"
    )

    if check_password():

        st.success("Acceso concedido.")
        st.write("Selecciona una empresa para cargar su modelo de clasificación y luego sube un archivo para procesar.")
        st.markdown("---")

        try:
            lista_archivos = [f for f in os.listdir('datos_empresas') if f.endswith('.xlsx')]
            nombres_empresas = {
                archivo: " ".join(word.capitalize() for word in archivo.replace('.xlsx', '').split('_'))
                for archivo in lista_archivos
            }
        except FileNotFoundError:
            st.error("Error crítico: No se encontró la carpeta 'datos_empresas'.")
            lista_archivos = []

        if not lista_archivos:
            st.warning("No se encontraron archivos de entrenamiento en la carpeta 'datos_empresas'.")
        else:
            archivo_seleccionado = st.selectbox(
                "Selecciona la Empresa",
                options=lista_archivos,
                format_func=lambda x: nombres_empresas[x]
            )

            if archivo_seleccionado:
                ruta_completa_archivo = os.path.join('datos_empresas', archivo_seleccionado)
                modelo_activo = cargar_y_entrenar_modelo(ruta_completa_archivo)

                if modelo_activo:
                    st.header("Cargar archivo para clasificar")
                    uploaded_file = st.file_uploader("Elige un archivo Excel (.xlsx)", type="xlsx", key=archivo_seleccionado)

                    if uploaded_file:
                        df_a_clasificar = pd.read_excel(uploaded_file)
                        st.write("Previsualización de los datos cargados:")
                        st.dataframe(df_a_clasificar.head())

                        nombre_columna = st.text_input("Escribe el nombre de la columna con las descripciones (ej: Glosa):")

                        if st.button('Clasificar Archivo'):
                            if nombre_columna and nombre_columna in df_a_clasificar.columns:
                                with st.spinner('Clasificando...'):
                                    try:
                                        df_clasificado = clasificar_datos(modelo_activo, df_a_clasificar, nombre_columna)
                                        st.success("¡Clasificación completada!")
                                        st.dataframe(df_clasificado.head())

                                        output = io.BytesIO()
                                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                            df_clasificado.to_excel(writer, index=False, sheet_name='Clasificaciones')

                                        st.download_button(
                                            label="📥 Descargar Resultado",
                                            data=output.getvalue(),
                                            file_name=f"resultado_{archivo_seleccionado}",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    except Exception as e:
                                        st.error(f"Error durante la clasificación: {e}")
                            else:
                                st.error(f"La columna '{nombre_columna}' no se encuentra en el archivo.")

if __name__ == "__main__":
    main()
