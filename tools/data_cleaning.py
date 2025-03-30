import pandas as pd
import numpy as np

def clean_data(df, operations=None):
    """
    Clean a DataFrame based on specified operations.
    
    Args:
        df (DataFrame): The pandas DataFrame to clean
        operations (dict or str, optional): Dictionary of cleaning operations to perform
                                    Keys can include: 'dropna', 'fillna', 'drop_duplicates',
                                    'convert_types', 'rename_columns', 'drop_columns'
                                    
    Returns:
        tuple: (cleaned DataFrame, str message)
    """
    if df is None:
        return None, "Error: No DataFrame provided"
    
    # Handle operations if it's a string representation of a dictionary
    if isinstance(operations, str):
        try:
            import ast
            # Use ast.literal_eval to safely convert the string to a dictionary
            operations = ast.literal_eval(operations)
        except:
            return None, f"Error: Could not parse operations: {operations}"
    
    if operations is None:
        operations = {
            'dropna': False,
            'fillna': {},
            'drop_duplicates': False,
            'convert_types': {},
            'rename_columns': {},
            'drop_columns': []
        }
    
    # Ensure operations is a dictionary
    if not isinstance(operations, dict):
        return None, f"Error: Operations must be a dictionary, got {type(operations)}"
    
    result_df = df.copy()
    message_parts = []
    
    # Drop rows with NaN values
    if operations.get('dropna', False):
        initial_rows = len(result_df)
        result_df = result_df.dropna()
        dropped_rows = initial_rows - len(result_df)
        message_parts.append(f"Dropped {dropped_rows} rows with NaN values")
    
    # Fill NaN values
    if operations.get('fillna'):
        for col, value in operations['fillna'].items():
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(value)
                message_parts.append(f"Filled NaN values in column '{col}'")
    
    # Drop duplicate rows
    if operations.get('drop_duplicates', False):
        initial_rows = len(result_df)
        result_df = result_df.drop_duplicates()
        dropped_rows = initial_rows - len(result_df)
        message_parts.append(f"Dropped {dropped_rows} duplicate rows")
    
    # Convert column data types
    if operations.get('convert_types'):
        for col, dtype in operations['convert_types'].items():
            if col in result_df.columns:
                try:
                    result_df[col] = result_df[col].astype(dtype)
                    message_parts.append(f"Converted '{col}' to {dtype}")
                except Exception as e:
                    message_parts.append(f"Failed to convert '{col}' to {dtype}: {str(e)}")
    
    # Rename columns
    if operations.get('rename_columns'):
        result_df = result_df.rename(columns=operations['rename_columns'])
        message_parts.append(f"Renamed {len(operations['rename_columns'])} columns")
    
    # Drop columns
    if operations.get('drop_columns'):
        cols_to_drop = [col for col in operations['drop_columns'] if col in result_df.columns]
        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)
            message_parts.append(f"Dropped columns: {', '.join(cols_to_drop)}")
    
    # Create a summary message
    if not message_parts:
        message = "No cleaning operations performed"
    else:
        message = "; ".join(message_parts)
    
    return result_df, message

def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Identify or handle outliers in specified columns.
    
    Args:
        df (DataFrame): The pandas DataFrame to process
        columns (list, optional): List of columns to check for outliers. If None, checks all numeric columns.
        method (str, optional): Method to identify outliers ('iqr', 'zscore', 'percentile')
        threshold (float, optional): Threshold for outlier detection
                                     For 'iqr': typically 1.5
                                     For 'zscore': typically 3
                                     For 'percentile': between 0 and 1, e.g., 0.01 for 1st percentile
    
    Returns:
        tuple: (DataFrame with outlier info, str message)
    """
    if df is None:
        return None, "Error: No DataFrame provided"
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to only include numeric columns from the provided list
        columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    if not columns:
        return df, "No numeric columns available for outlier detection"
    
    result_df = df.copy()
    outlier_counts = {}
    
    for col in columns:
        is_outlier = np.zeros(len(df), dtype=bool)
        
        # IQR method
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Z-score method
        elif method == 'zscore':
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            is_outlier = abs(z_scores) > threshold
        
        # Percentile method
        elif method == 'percentile':
            lower_bound = df[col].quantile(threshold)
            upper_bound = df[col].quantile(1 - threshold)
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Count outliers
        outlier_counts[col] = is_outlier.sum()
        
        # Add outlier flag column
        result_df[f'{col}_is_outlier'] = is_outlier
    
    message = f"Identified outliers using {method} method: " + ", ".join([f"{col}: {count}" for col, count in outlier_counts.items()])
    return result_df, message

def examine_dataframe(df):
    """
    Examine the structure of a DataFrame.
    
    Args:
        df (DataFrame): The pandas DataFrame to examine
        
    Returns:
        tuple: (dict of DataFrame info, str message)
    """
    if df is None:
        return None, "Error: No DataFrame provided"
    
    info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'head': df.head(5).to_dict(orient='records')
    }
    
    message = f"DataFrame has {df.shape[0]} rows and {df.shape[1]} columns: {', '.join(df.columns.tolist())}"
    return info, message 