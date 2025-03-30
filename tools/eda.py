import pandas as pd
import numpy as np
import json

def compute_statistics(df, columns=None):
    """
    Compute summary statistics for the given DataFrame.
    
    Args:
        df (DataFrame): The pandas DataFrame to analyze
        columns (list or str, optional): List of columns to analyze. If None, analyzes all columns.
        
    Returns:
        tuple: (dict of statistics, str message)
    """
    if df is None or df.empty:
        return None, "Error: DataFrame is empty or None"
    
    # Handle columns parameter if it's a string
    if isinstance(columns, str):
        # If it looks like a list representation, try to convert it
        if columns.startswith('[') and columns.endswith(']'):
            try:
                import ast
                columns = ast.literal_eval(columns)
            except:
                # If we can't convert it and it's not a column name, use all columns
                if columns not in df.columns:
                    columns = None
        # If it's a single column name as string, convert to list
        elif columns in df.columns:
            columns = [columns]
        else:
            return None, f"Error: Column '{columns}' not found. Available columns: {', '.join(df.columns.tolist())}"
    
    if columns is None:
        columns = df.columns.tolist()
    else:
        # Validate each column exists in the DataFrame
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            available_cols = df.columns.tolist()
            return None, f"Error: Columns {invalid_columns} not found. Available columns: {', '.join(available_cols)}"
            
        # Filter to include only columns that exist in the DataFrame
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        available_cols = df.columns.tolist()
        return None, f"Error: No valid columns specified. Available columns: {', '.join(available_cols)}"
    
    stats = {}
    
    # Get basic info
    stats['row_count'] = len(df)
    stats['column_count'] = len(df.columns)
    
    # Compute missing values counts and percentages
    missing_values = df[columns].isnull().sum().to_dict()
    missing_percentages = (df[columns].isnull().mean() * 100).to_dict()
    
    stats['missing_values'] = {
        col: {
            'count': int(missing_values[col]),
            'percentage': round(missing_percentages[col], 2)
        } for col in columns
    }
    
    # Compute data types
    stats['data_types'] = {col: str(df[col].dtype) for col in columns}
    
    # Compute numeric statistics for numeric columns
    numeric_columns = df[columns].select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        numeric_stats = df[numeric_columns].describe().to_dict()
        stats['numeric_stats'] = numeric_stats
    
    # Compute categorical statistics for non-numeric columns
    categorical_columns = [col for col in columns if col not in numeric_columns]
    if categorical_columns:
        stats['categorical_stats'] = {}
        for col in categorical_columns:
            value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 most common values
            unique_count = df[col].nunique()
            stats['categorical_stats'][col] = {
                'unique_count': unique_count,
                'top_values': value_counts
            }
    
    message = f"Computed statistics for {len(columns)} columns ({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)"
    return stats, message

def compute_correlations(df, method='pearson'):
    """
    Compute correlation matrix for numeric columns.
    
    Args:
        df (DataFrame): The pandas DataFrame to analyze
        method (str, optional): Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        tuple: (correlation DataFrame, str message)
    """
    if df is None or df.empty:
        return None, "Error: DataFrame is empty or None"
    
    # Handle method parameter if it's quoted
    if isinstance(method, str):
        # Remove quotes if present
        method = method.strip("'\"")
        
    # Validate the method
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        return None, f"Error: Invalid correlation method '{method}'. Valid options are: {', '.join(valid_methods)}"
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None, "Error: Not enough numeric columns for correlation analysis"
    
    try:
        # Calculate correlations
        corr_matrix = numeric_df.corr(method=method)
        message = f"Computed {method} correlations for {len(numeric_df.columns)} numeric columns"
        return corr_matrix, message
    except Exception as e:
        return None, f"Error computing correlations: {str(e)}"

def identify_important_features(df, target_column, method='correlation'):
    """
    Identify potentially important features for a target column.
    
    Args:
        df (DataFrame): The pandas DataFrame to analyze
        target_column (str): The target column to analyze against
        method (str, optional): Method to identify importance ('correlation')
        
    Returns:
        tuple: (dict of important features, str message)
    """
    if df is None or df.empty:
        return None, "Error: DataFrame is empty or None"
    
    if target_column not in df.columns:
        return None, f"Error: Target column '{target_column}' not found in DataFrame"
    
    # Check if target column is numeric
    if not np.issubdtype(df[target_column].dtype, np.number):
        return None, f"Error: Target column '{target_column}' must be numeric for this analysis"
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    if len(numeric_df.columns) < 2:
        return None, "Error: Not enough numeric columns for feature importance analysis"
    
    if method == 'correlation':
        # Calculate correlations with target column
        correlations = numeric_df.corr()['target_column'] if target_column not in numeric_df.columns else numeric_df.corr()[target_column]
        # Drop the correlation with itself
        correlations = correlations.drop(target_column) if target_column in correlations.index else correlations
        
        # Sort by absolute correlation values
        abs_corr = correlations.abs().sort_values(ascending=False)
        important_features = abs_corr.to_dict()
        
        message = f"Identified feature importance based on correlation with '{target_column}'"
        return important_features, message
    else:
        return None, f"Error: Unsupported method '{method}'" 