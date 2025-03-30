import pandas as pd
import logging

# Configurar logger
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocessing function with feature engineering and robustness improvements.
    
    Parameters:
    -----------
    df : DataFrame
        Input data frame
        
    Returns:
    --------
    DataFrame
        Processed data frame with engineered features
    """
    logger.info("Starting data preprocessing...")

    # Garantir que todas as colunas esperadas existem, preenchendo com NaN se necessário
    expected_columns = [
        'CCS Procedure Code', 'Other Provider License Number', 'Race',
        'APR Severity of Illness Code', 'APR Risk of Mortality',
        'Facility Id', 'Age Group', 'CCS Diagnosis Code', 'Length of Stay'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Criar variáveis derivadas
    df['Procedure_Performed'] = (df['CCS Procedure Code'] != 0).astype(int)
    df['Multiple_Providers'] = (~df['Other Provider License Number'].isna()).astype(int)

    # Normalizar categorias de raça
    df['Race'] = df['Race'].astype(str).str.strip().replace({
        'Black/African American': 'Black',
        'White': 'White',
        'Other Race': 'Other',
        'Multi-racial': 'Multiracial'
    }).fillna('Unknown')

    # Criar Combined_Severity
    severity_map = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
    mortality_map = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}

    df['APR Severity of Illness Code'] = df['APR Severity of Illness Code'].map(severity_map).fillna(0)
    df['APR Risk of Mortality'] = df['APR Risk of Mortality'].map(mortality_map).fillna(0)
    df['Combined_Severity'] = df['APR Severity of Illness Code'] * df['APR Risk of Mortality']

    # Garantir que CCS Diagnosis Code seja numérico antes de qualquer operação
    df['CCS Diagnosis Code'] = pd.to_numeric(df['CCS Diagnosis Code'], errors='coerce').fillna(0).astype(int)

    # Criar média da estadia por Facility Id
    if 'Facility Id' in df.columns and 'Length of Stay' in df.columns:
        facility_means = df.groupby('Facility Id')['Length of Stay'].mean().to_dict()
        df['Facility_Avg_LOS'] = df['Facility Id'].map(lambda x: facility_means.get(x, df['Length of Stay'].mean()))

    # Criar Age_Numeric
    age_order = ['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older']
    age_map = {age: i for i, age in enumerate(age_order)}
    df['Age_Numeric'] = df['Age Group'].map(age_map).fillna(-1)

    # Criar interação idade-severidade
    df['Age_Severity_Interaction'] = df['Age_Numeric'] * df['Combined_Severity']

    # Criar Diagnosis_Category e a média da estadia por categoria
    df['Diagnosis_Category'] = (df['CCS Diagnosis Code'] // 100).fillna(-1).astype(int)
    diag_means = df.groupby('Diagnosis_Category')['Length of Stay'].mean().to_dict()
    df['Diagnosis_Category_Avg_LOS'] = df['Diagnosis_Category'].map(lambda x: diag_means.get(x, df['Length of Stay'].mean()))

    # Log de validação
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Race distribution: {df['Race'].value_counts().to_dict()}")

    return df
