#!/usr/bin/env python3
"""
Example script demonstrating how to use the autonomous data scientist agent programmatically.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from main_controller import DataScientistAgent

# Load environment variables
load_dotenv()

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

def run_example(debug=True):
    """Run an example analysis using the agent."""
    # Create sample data if it doesn't exist
    excel_path = "data/sales_2023.xlsx"
    if not os.path.exists(excel_path):
        excel_path = create_sample_data()
    
    # Initialize the agent
    agent = DataScientistAgent(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define a task
    task = f"Analyze sales data from {excel_path}. Perform the following steps:\n" \
           f"1. Load and examine the data structure to understand available columns\n" \
           f"2. Clean the data by handling missing values\n" \
           f"3. Generate summary statistics for each product type\n" \
           f"4. Create visualizations for monthly sales trends and product comparison\n" \
           f"5. Identify top-performing products and regions\n" \
           f"6. Generate an HTML report with all findings\n\n" \
           f"IMPORTANT: When working with DataFrames, always use 'df' (not in quotes) to refer to the current DataFrame." \
           f"After loading data, always examine its structure with examine_dataframe(df=df) before performing operations."
    
    print(f"\nSubmitting task to agent: {task}\n")
    
    if debug:
        print("DEBUG MODE: Running the direct tool usage example instead of the ReAct agent")
        # Run direct tool usage for simplicity
        direct_tool_usage_example()
        return
    
    # Run the agent
    result, log = agent.run(task, max_iterations=15)
    
    print("\n=== Task Complete ===")
    print(result)
    
    # Print only the actions taken
    print("\n=== Actions Taken ===")
    for entry in log:
        if entry.startswith("Action:"):
            print(entry)

def direct_tool_usage_example():
    """Example of directly using the agent's tools without the ReAct loop."""
    # Create sample data if it doesn't exist
    excel_path = "data/sales_2023.xlsx"
    if not os.path.exists(excel_path):
        excel_path = create_sample_data()
    
    # Initialize the agent
    agent = DataScientistAgent()
    
    print("\n=== Direct Tool Usage Example ===")
    
    # Read the Excel file
    print("Reading Excel file...")
    df, message = agent.execute_action("read_excel", file_path=excel_path)
    print(message)
    
    if df is not None:
        # Clean the data
        print("\nCleaning data...")
        clean_ops = {
            'dropna': False,
            'fillna': {'Customer_Age': df['Customer_Age'].median(), 
                      'Customer_Gender': 'Unknown'},
            'drop_duplicates': True
        }
        df_clean, message = agent.execute_action("clean_data", df=df, operations=clean_ops)
        print(message)
        
        # Compute statistics
        print("\nComputing statistics...")
        stats, message = agent.execute_action("compute_statistics", df=df_clean)
        print(message)
        
        # Generate a visualization
        print("\nGenerating visualization...")
        # Group by product and sum sales
        product_sales = df_clean.groupby('Product')['Total_Sales'].sum().reset_index()
        plot_path, message = agent.execute_action(
            "generate_plot", 
            df=product_sales, 
            plot_type="bar", 
            x="Product", 
            y="Total_Sales", 
            title="Total Sales by Product",
            xlabel="Product",
            ylabel="Total Sales ($)"
        )
        print(message)
        
        # Generate a report
        print("\nGenerating report...")
        # Create a summary
        summary = "This report analyzes sales data for 2023. "
        summary += f"The dataset contains {len(df_clean)} records after cleaning. "
        summary += f"The total sales amount was ${df_clean['Total_Sales'].sum():,.2f}. "
        summary += f"The best-selling product was {product_sales.loc[product_sales['Total_Sales'].idxmax(), 'Product']}."
        
        # Generate report with plot
        report_path, message = agent.execute_action(
            "generate_report",
            title="Sales Analysis Report - 2023",
            summary=summary,
            plots=[plot_path] if plot_path else None,
            dataframes={"Product Sales Summary": product_sales},
            format="html"
        )
        print(message)

if __name__ == "__main__":
    # Ensure data directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Make sure we have sample data
    if not os.path.exists("data/sales_2023.xlsx"):
        create_sample_data()
    
    # Check if we should run in non-debug mode
    import sys
    debug_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-debug':
        debug_mode = False
    
    # Run the example
    run_example(debug=debug_mode)
    
    # If you want to run the direct tool usage example separately
    # direct_tool_usage_example() 