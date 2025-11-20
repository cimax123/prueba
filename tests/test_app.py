import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import io
import os
from app_multiempresa import check_password, cargar_y_entrenar_modelo, clasificar_datos

# --- Tests for check_password ---

@patch('streamlit.secrets', new_callable=MagicMock)
@patch('streamlit.text_input')
@patch('streamlit.stop')
@patch('streamlit.error')
@patch('streamlit.warning')
@patch('streamlit.title')
def test_check_password_correct(mock_title, mock_warning, mock_error, mock_stop, mock_text_input, mock_secrets):
    """Test that check_password returns True with correct password."""
    mock_secrets.__getitem__.return_value = "secret123"
    mock_text_input.return_value = "secret123"

    assert check_password() is True
    mock_stop.assert_not_called()
    mock_error.assert_not_called()

@patch('streamlit.secrets', new_callable=MagicMock)
@patch('streamlit.text_input')
@patch('streamlit.stop')
@patch('streamlit.error')
@patch('streamlit.warning')
@patch('streamlit.title')
def test_check_password_incorrect(mock_title, mock_warning, mock_error, mock_stop, mock_text_input, mock_secrets):
    """Test that check_password stops and shows error with incorrect password."""
    mock_secrets.__getitem__.return_value = "secret123"
    mock_text_input.return_value = "wrongpass"

    # Since check_password calls st.stop() which raises a StopException in streamlit but is mocked here,
    # we just check if it returns None (default return of function without explicit return) or stops execution
    assert check_password() is None
    mock_error.assert_called_with("La contraseña es incorrecta.")
    mock_stop.assert_called()

@patch('streamlit.secrets', new_callable=MagicMock)
@patch('streamlit.text_input')
@patch('streamlit.stop')
@patch('streamlit.title')
def test_check_password_empty(mock_title, mock_stop, mock_text_input, mock_secrets):
    """Test that check_password stops if password is empty."""
    mock_secrets.__getitem__.return_value = "secret123"
    mock_text_input.return_value = ""

    assert check_password() is None
    mock_stop.assert_called()

@patch('streamlit.secrets', new_callable=MagicMock)
@patch('streamlit.text_input')
@patch('streamlit.stop')
@patch('streamlit.warning')
@patch('streamlit.title')
def test_check_password_missing_secrets(mock_title, mock_warning, mock_stop, mock_text_input, mock_secrets):
    """Test fallback to default password when secrets are missing."""
    mock_secrets.__getitem__.side_effect = KeyError # Simulate KeyError on dictionary access
    mock_text_input.return_value = "test"

    assert check_password() is True
    mock_warning.assert_called()
    mock_stop.assert_not_called()

# --- Tests for cargar_y_entrenar_modelo ---

@pytest.fixture
def valid_training_file(tmp_path):
    """Creates a temporary Excel file with valid training data."""
    data = {
        'numero_cuenta': [1, 2, 3, 4, 5],
        'descripcion': ['compra oficina', 'venta productos', 'pago luz', 'compra sillas', 'venta servicios'],
        'nombre_cuenta': ['gastos', 'ingresos', 'gastos', 'gastos', 'ingresos']
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "training_data.xlsx"
    df.to_excel(filepath, header=False, index=False)
    return str(filepath)

@pytest.fixture
def invalid_training_file(tmp_path):
    """Creates a temporary Excel file with insufficient training data."""
    data = {
        'numero_cuenta': [1],
        'descripcion': ['compra oficina'],
        'nombre_cuenta': ['gastos']
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "insufficient_data.xlsx"
    df.to_excel(filepath, header=False, index=False)
    return str(filepath)

@patch('streamlit.cache_resource', lambda x: x) # Mock the cache decorator
@patch('streamlit.error')
def test_cargar_y_entrenar_modelo_success(mock_error, valid_training_file):
    """Test successful model training."""
    model = cargar_y_entrenar_modelo(valid_training_file)
    assert model is not None
    # Check if model has predict method
    assert hasattr(model, 'predict')

@patch('streamlit.cache_resource', lambda x: x)
@patch('streamlit.error')
def test_cargar_y_entrenar_modelo_insufficient_data(mock_error, invalid_training_file):
    """Test model training with insufficient data."""
    model = cargar_y_entrenar_modelo(invalid_training_file)
    assert model is None
    mock_error.assert_called()
    assert "no tiene suficientes datos" in mock_error.call_args[0][0]

@patch('streamlit.cache_resource', lambda x: x)
@patch('streamlit.error')
def test_cargar_y_entrenar_modelo_file_not_found(mock_error):
    """Test handling of missing file."""
    model = cargar_y_entrenar_modelo("non_existent_file.xlsx")
    assert model is None
    mock_error.assert_called()

# --- Tests for clasificar_datos ---

def test_clasificar_datos_success():
    """Test that clasificar_datos correctly adds predictions."""
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = ['gastos', 'ingresos']

    df = pd.DataFrame({
        'Glosa': ['compra papel', 'venta consultoria']
    })

    result_df = clasificar_datos(mock_model, df, 'Glosa')

    assert 'cuenta_sugerida' in result_df.columns
    assert result_df['cuenta_sugerida'].tolist() == ['gastos', 'ingresos']
    mock_model.predict.assert_called_once()

def test_clasificar_datos_missing_column():
    """Test that clasificar_datos raises KeyError if column is missing."""
    mock_model = MagicMock()
    df = pd.DataFrame({
        'Glosa': ['compra papel']
    })

    with pytest.raises(KeyError):
        clasificar_datos(mock_model, df, 'Descripcion')

def test_clasificar_datos_prediction_error():
    """Test that clasificar_datos propagates exceptions from model prediction."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("Prediction failed")

    df = pd.DataFrame({
        'Glosa': ['compra papel']
    })

    with pytest.raises(ValueError, match="Prediction failed"):
        clasificar_datos(mock_model, df, 'Glosa')
