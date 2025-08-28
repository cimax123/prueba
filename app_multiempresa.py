# -*- coding: utf-8 -*-

# --- 1. IMPORTACIONES: Todas las librerías van juntas al principio ---
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import os
import io

# --- 2. CONFIGURACIÓN DE PÁGINA: Debe ser el primer comando de Streamlit ---
st.set_page_config(
    page_title="Asistente Contable Multi-Empresa",
    layout="wide"
)

# --- 3. DEFINICIÓN DE FUNCIONES ---

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

def get_corrected_data_path(file_name):
    """Genera la ruta del archivo de datos corregidos."""
    name_without_ext = os.path.splitext(file_name)[0]
    return f"datos_empresas/corregidos_{name_without_ext}.xlsx"

@st.cache_resource
def cargar_y_entrenar_modelo(ruta_archivo_empresa, ruta_archivo_corregido):
    try:
        # Cargar los datos de entrenamiento originales
        df_train = pd.read_excel(ruta_archivo_empresa, header=None, names=['numero_cuenta', 'descripcion', 'nombre_cuenta'])
        df_train.dropna(subset=['descripcion', 'nombre_cuenta'], inplace=True)
        
        # Cargar datos corregidos si existen y combinarlos
        if os.path.exists(ruta_archivo_corregido):
            df_corregido = pd.read_excel(ruta_archivo_corregido, index_col=None)
            df_corregido = df_corregido[['descripcion', 'cuenta_corregida']].copy()
            df_corregido.rename(columns={'cuenta_corregida': 'nombre_cuenta'}, inplace=True)
            df_combined = pd.concat([df_train[['descripcion', 'nombre_cuenta']], df_corregido], ignore_index=True)
            df_combined.dropna(subset=['descripcion', 'nombre_cuenta'], inplace=True)
            df_combined['descripcion'] = df_combined['descripcion'].astype(str).str.lower()
            df_combined.drop_duplicates(subset=['descripcion'], keep='last', inplace=True)
        else:
            df_combined = df_train
            df_combined['descripcion'] = df_combined['descripcion'].astype(str).str.lower()
            
        counts = df_combined['nombre_cuenta'].value_counts()
        df_filtrado = df_combined[df_combined['nombre_cuenta'].isin(counts[counts >= 2].index)]
        
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

# --- 4. LÓGICA PRINCIPAL DE LA APLICACIÓN ---
if check_password():
    st.success("Acceso concedido.")
    st.write("Selecciona una empresa para cargar su modelo, sube un archivo para procesar y, si lo deseas, corrige las clasificaciones para que el modelo aprenda.")
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
        st.session_state.archivo_seleccionado = archivo_seleccionado

        if archivo_seleccionado:
            ruta_completa_archivo = os.path.join('datos_empresas', archivo_seleccionado)
            ruta_archivo_corregido = get_corrected_data_path(archivo_seleccionado)
            
            # Use a unique key for the cache resource to force a re-run on selection change
            modelo_activo = cargar_y_entrenar_modelo(ruta_completa_archivo, ruta_archivo_corregido)

            if modelo_activo:
                st.header("Cargar archivo para clasificar")
                uploaded_file = st.file_uploader("Elige un archivo Excel (.xlsx)", type="xlsx", key="uploader_" + archivo_seleccionado)

                if uploaded_file:
                    df_a_clasificar = pd.read_excel(uploaded_file)
                    st.session_state.df_original = df_a_clasificar.copy()
                    st.write("Previsualización de los datos cargados:")
                    st.dataframe(df_a_clasificar.head())
                    
                    nombre_columna = st.text_input("Escribe el nombre de la columna con las descripciones (ej: Glosa):")
                    
                    if st.button('Clasificar y Corregir'):
                        if nombre_columna and nombre_columna in df_a_clasificar.columns:
                            with st.spinner('Clasificando...'):
                                descripciones = df_a_clasificar[nombre_columna].astype(str).str.lower()
                                predicciones = modelo_activo.predict(descripciones)
                                df_a_clasificar['cuenta_sugerida'] = predicciones
                                df_a_clasificar['cuenta_corregida'] = df_a_clasificar['cuenta_sugerida']
                                
                                st.success("¡Clasificación completada! Ahora puedes corregir las sugerencias.")
                                st.session_state.df_clasificado = df_a_clasificar
                                
                                # Ofrecer el archivo para descargar
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df_a_clasificar.to_excel(writer, index=False, sheet_name='Clasificaciones')
                                
                                st.download_button(
                                    label="📥 Descargar Resultado",
                                    data=output.getvalue(),
                                    file_name=f"resultado_{archivo_seleccionado}",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                    # Sección de edición y re-entrenamiento
                    if 'df_clasificado' in st.session_state:
                        st.markdown("---")
                        st.header("Corrige las clasificaciones y guarda los cambios")
                        edited_df = st.data_editor(
                            st.session_state.df_clasificado,
                            column_config={
                                "cuenta_corregida": st.column_config.SelectboxColumn(
                                    "Cuenta Corregida",
                                    help="Selecciona la cuenta correcta",
                                    options=modelo_activo.classes_.tolist(),
                                    required=True,
                                )
                            },
                            num_rows="dynamic"
                        )
                        st.session_state.edited_df = edited_df

                        if st.button('Guardar Correcciones y Re-entrenar'):
                            with st.spinner('Guardando y re-entrenando...'):
                                edited_data = st.session_state.edited_df
                                edited_data = edited_data.dropna(subset=[nombre_columna, 'cuenta_corregida'])
                                edited_data = edited_data.rename(columns={nombre_columna: 'descripcion'})
                                
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    edited_data[['descripcion', 'cuenta_corregida']].to_excel(writer, index=False)
                                
                                # Guardar en disco
                                with open(ruta_archivo_corregido, "wb") as f:
                                    f.write(output.getvalue())
                                
                                # Forzar el re-entrenamiento del modelo
                                st.cache_resource.clear()
                                st.success("¡Correcciones guardadas! El modelo será re-entrenado con estos nuevos datos.")
                                st.rerun()

                        # Ofrecer descarga de las correcciones guardadas
                        if os.path.exists(ruta_archivo_corregido):
                            with open(ruta_archivo_corregido, "rb") as f:
                                st.download_button(
                                    label="💾 Descargar Datos Corregidos",
                                    data=f,
                                    file_name=os.path.basename(ruta_archivo_corregido),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        else:
                            st.info("Aún no hay correcciones guardadas para esta empresa.")
                        
                        
                        
