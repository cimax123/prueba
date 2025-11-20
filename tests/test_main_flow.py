import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import io
import os
from app_multiempresa import check_password

@patch('streamlit.secrets', new_callable=MagicMock)
@patch('streamlit.text_input')
@patch('streamlit.stop')
@patch('streamlit.warning')
@patch('streamlit.title')
def test_check_password_file_not_found(mock_title, mock_warning, mock_stop, mock_text_input, mock_secrets):
    """Test fallback to default password when secrets file is not found."""
    mock_secrets.__getitem__.side_effect = FileNotFoundError
    mock_text_input.return_value = "test"

    assert check_password() is True
    mock_warning.assert_called()

@patch('app_multiempresa.check_password')
@patch('os.listdir')
@patch('streamlit.error')
@patch('streamlit.warning')
def test_main_no_files(mock_warning, mock_error, mock_listdir, mock_check_password):
    """Test main function when no excel files are found."""
    mock_check_password.return_value = True
    mock_listdir.return_value = []

    from app_multiempresa import main
    main()

    mock_warning.assert_called_with("No se encontraron archivos de entrenamiento en la carpeta 'datos_empresas'.")

@patch('app_multiempresa.check_password')
@patch('os.listdir')
@patch('streamlit.error')
def test_main_folder_not_found(mock_error, mock_listdir, mock_check_password):
    """Test main function when 'datos_empresas' folder is missing."""
    mock_check_password.return_value = True
    mock_listdir.side_effect = FileNotFoundError

    from app_multiempresa import main
    main()

    mock_error.assert_called_with("Error crítico: No se encontró la carpeta 'datos_empresas'.")

@patch('app_multiempresa.check_password')
@patch('os.listdir')
@patch('streamlit.selectbox')
@patch('app_multiempresa.cargar_y_entrenar_modelo')
@patch('streamlit.file_uploader')
def test_main_flow_success(mock_file_uploader, mock_cargar, mock_selectbox, mock_listdir, mock_check_password):
    """Test full successful flow in main."""
    mock_check_password.return_value = True
    mock_listdir.return_value = ['empresa1.xlsx']
    mock_selectbox.return_value = 'empresa1.xlsx'

    # Mock loaded model
    mock_model = MagicMock()
    mock_cargar.return_value = mock_model

    # Mock uploaded file
    mock_file_uploader.return_value = io.BytesIO(b"fake excel content")

    # Mock dataframe loading and classification
    with patch('pandas.read_excel') as mock_read_excel, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.button') as mock_button, \
         patch('app_multiempresa.clasificar_datos') as mock_clasificar, \
         patch('streamlit.download_button') as mock_download:

        # Setup mock dataframe
        mock_df = pd.DataFrame({'Glosa': ['test']})
        mock_read_excel.return_value = mock_df

        # User inputs column name and clicks button
        mock_text_input.return_value = 'Glosa'
        mock_button.return_value = True

        mock_clasificar.return_value = mock_df # Return same df

        from app_multiempresa import main
        main()

        mock_clasificar.assert_called()
        mock_download.assert_called()

@patch('app_multiempresa.check_password')
@patch('os.listdir')
@patch('streamlit.selectbox')
@patch('app_multiempresa.cargar_y_entrenar_modelo')
@patch('streamlit.file_uploader')
def test_main_flow_bad_column(mock_file_uploader, mock_cargar, mock_selectbox, mock_listdir, mock_check_password):
    """Test flow where user enters invalid column name."""
    mock_check_password.return_value = True
    mock_listdir.return_value = ['empresa1.xlsx']
    mock_selectbox.return_value = 'empresa1.xlsx'
    mock_cargar.return_value = MagicMock()
    mock_file_uploader.return_value = io.BytesIO(b"fake")

    with patch('pandas.read_excel') as mock_read_excel, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.error') as mock_error:

        mock_df = pd.DataFrame({'Glosa': ['test']})
        mock_read_excel.return_value = mock_df

        # Invalid column
        mock_text_input.return_value = 'WrongColumn'
        mock_button.return_value = True

        from app_multiempresa import main
        main()

        mock_error.assert_called()
        assert "no se encuentra en el archivo" in mock_error.call_args[0][0]
