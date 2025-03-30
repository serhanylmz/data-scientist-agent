import pandas as pd
from sqlalchemy import create_engine
import logging
import os

def read_excel(file_path, sheet_name=0):
    """
    Read data from an Excel file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str or int, optional): Sheet name or index. Defaults to 0 (first sheet).
        
    Returns:
        tuple: (DataFrame, str message)
    """
    try:
        if not os.path.exists(file_path):
            return None, f"Error: File {file_path} does not exist"
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df, f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns"
    except Exception as e:
        return None, f"Error loading Excel file: {str(e)}"

def read_sql(query, connection_string="sqlite:///data/default.db"):
    """
    Read data from a SQL database using a query.
    
    Args:
        query (str): SQL query to execute
        connection_string (str, optional): Database connection string. 
                                          Defaults to local SQLite database.
        
    Returns:
        tuple: (DataFrame, str message)
    """
    try:
        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return df, f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns from database"
    except Exception as e:
        return None, f"Error executing SQL query: {str(e)}"

def list_excel_sheets(file_path):
    """
    List all sheets in an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        tuple: (list of sheet names, str message)
    """
    try:
        if not os.path.exists(file_path):
            return None, f"Error: File {file_path} does not exist"
        
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        return sheet_names, f"Found {len(sheet_names)} sheets in the Excel file"
    except Exception as e:
        return None, f"Error listing Excel sheets: {str(e)}" 