#!/usr/bin/env python3
"""
Script to create sample sales data for testing the autonomous data scientist agent.
"""

import os
import pandas as pd
import numpy as np

def create_sample_data():
    """Create a sample Excel file for demonstration."""
    print("Creating sample data...")
    
    # Create a 'data' directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create sample sales data
    np.random.seed(42)  # For reproducibility
    
    # Date range for 2023
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Product categories
    products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Headphones']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate random data
    n_records = 1000
    
    data = {
        'Date': np.random.choice(dates, n_records),
        'Product': np.random.choice(products, n_records),
        'Region': np.random.choice(regions, n_records),
        'Units_Sold': np.random.randint(1, 50, n_records),
        'Unit_Price': np.random.uniform(100, 1500, n_records).round(2),
        'Customer_Age': np.random.randint(18, 70, n_records),
        'Customer_Gender': np.random.choice(['M', 'F'], n_records),
        'Customer_ID': [f'CUST{i:04d}' for i in range(1, n_records + 1)]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Add a column for total sales
    df['Total_Sales'] = df['Units_Sold'] * df['Unit_Price']
    
    # Add some missing values
    indices = np.random.choice(n_records, 50, replace=False)
    df.loc[indices, 'Customer_Age'] = np.nan
    
    indices = np.random.choice(n_records, 50, replace=False)
    df.loc[indices, 'Customer_Gender'] = np.nan
    
    # Save to Excel
    excel_path = "data/sales_2023.xlsx"
    df.to_excel(excel_path, index=False)
    
    print(f"Sample data created: {excel_path}")
    return excel_path

if __name__ == "__main__":
    create_sample_data() 