# Autonomous Data Scientist Agent

An autonomous data scientist agent that uses the ReAct (Reasoning, Acting) paradigm to perform data analysis tasks. The agent can ingest data, clean it, perform exploratory data analysis (EDA), generate visualizations, and compile reports.

## Features

- **ReAct Paradigm**: Alternates between reasoning ("Thought"), acting ("Action"), and observing results ("Observation")
- **Data Ingestion**: Reads data from Excel files and SQL databases
- **Data Cleaning**: Handles missing values, removes duplicates, converts data types
- **Exploratory Data Analysis**: Computes statistics, identifies correlations, analyzes features
- **Visualization**: Generates various types of plots (line, bar, scatter, histogram, boxplot, heatmap)
- **Reporting**: Creates HTML or PDF reports with summaries and visualizations
- **LLM Integration**: Uses an LLM (e.g., GPT-4) for natural language reasoning

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd data-scientist-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Project Structure

```
data-scientist-agent/
│
├── main_controller.py      # Main controller implementing the ReAct loop
├── llm_interface.py        # Interface for LLM API calls
├── logging_module.py       # Logging module for tracking Thoughts, Actions, Observations
├── tools/
│   ├── __init__.py         # Package initialization
│   ├── data_ingestion.py   # Functions for reading data
│   ├── data_cleaning.py    # Functions for cleaning data
│   ├── eda.py              # Functions for exploratory data analysis
│   ├── visualization.py    # Functions for generating plots
│   └── reporting.py        # Functions for generating reports
├── logs/                   # Log files (created at runtime)
├── output/                 # Output files (created at runtime)
│   ├── plots/              # Generated plots
│   └── reports/            # Generated reports
└── requirements.txt        # Project dependencies
```

## Usage

Run the agent with a task description:

```bash
python main_controller.py "Analyze sales data from data/sales.xlsx and generate a report with visualizations"
```

Optional arguments:
- `--api-key`: Specify the API key (alternative to using a .env file)
- `--model`: Specify the model to use (default: gpt-4)
- `--max-iterations`: Maximum number of iterations (default: 10)

## Example Tasks

1. Basic data analysis:
   ```
   python main_controller.py "Load data/sales.xlsx, clean it, and generate summary statistics"
   ```

2. Complete analysis with visualizations:
   ```
   python main_controller.py "Analyze Q4 sales data from data/sales_q4.xlsx, identify trends, and generate a report with visualizations of the top-selling products"
   ```

3. SQL database analysis:
   ```
   python main_controller.py "Connect to the SQLite database at data/customers.db, analyze customer demographics, and create a PDF report with visualizations"
   ```

## Extending the Agent

The agent is designed to be modular and extensible:

1. Add new tool functions to the appropriate files in the `tools/` directory
2. The main controller will automatically detect and make these functions available to the agent

## Logs and Traceability

All agent operations are logged with timestamp, thought process, actions taken, and observations. This provides traceability and helps diagnose issues.

Logs are stored in two formats:
- Text logs in the `logs/` directory
- Structured JSON logs for machine processing

## License

[MIT License](LICENSE) 