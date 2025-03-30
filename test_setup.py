#!/usr/bin/env python3
"""
Test script to verify the setup of the autonomous data scientist agent.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is set up correctly."""
    print("Checking environment...")
    
    # Check Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API key: Found")
    else:
        print("OpenAI API key: Not found - please set OPENAI_API_KEY in .env file")
    
    # Check if required directories exist
    directories = ["data", "output/plots", "output/reports", "logs", "tools"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"Directory '{directory}': Found")
        else:
            print(f"Directory '{directory}': Not found - will be created")
            os.makedirs(directory, exist_ok=True)
    
    # Check if required modules are installed
    required_modules = ["pandas", "numpy", "matplotlib", "openai", "jinja2", "sqlalchemy", "openpyxl"]
    for module in required_modules:
        try:
            __import__(module)
            print(f"Module '{module}': Installed")
        except ImportError:
            print(f"Module '{module}': Not installed - please install with 'pip install {module}'")
    
    # Check for sample data
    if os.path.exists("data/sales_2023.xlsx"):
        print("Sample data: Found")
    else:
        print("Sample data: Not found - will be created when running example.py")
    
    print("\nSetup check complete.")

def test_openai_connection():
    """Test connection to OpenAI API."""
    print("\nTesting OpenAI connection...")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not found. Skipping connection test.")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # Simple test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Connection successful' if you receive this message."}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"OpenAI response: {result}")
        
        if "connection successful" in result.lower():
            print("OpenAI connection: Successful")
            return True
        else:
            print("OpenAI connection: Unexpected response")
            return False
            
    except Exception as e:
        print(f"OpenAI connection error: {str(e)}")
        return False

if __name__ == "__main__":
    # Check environment setup
    check_environment()
    
    # Test OpenAI connection
    test_openai_connection()
    
    print("\nIf all checks passed, you can now run the example script with: python example.py")
    print("If you encountered any issues, please fix them before proceeding.") 