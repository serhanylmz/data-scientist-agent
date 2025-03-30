import os
from datetime import datetime
import jinja2
import base64
import pandas as pd
import json

def generate_report(title, summary, plots=None, dataframes=None, output_dir='./output/reports', format='html'):
    """
    Generate a report from the analysis results.
    
    Args:
        title (str): Report title
        summary (str): Text summary of the analysis
        plots (list, optional): List of plot file paths to include in the report
        dataframes (dict, optional): Dictionary of DataFrames to include in the report, with keys as table names
        output_dir (str, optional): Directory to save the report
        format (str, optional): Report format ('html' or 'pdf')
        
    Returns:
        tuple: (report file path, str message)
    """
    # Handle string parameters
    if isinstance(title, str):
        title = title.strip("'\"")
    
    if isinstance(summary, str):
        summary = summary.strip("'\"")
    
    if isinstance(format, str):
        format = format.strip("'\"")
    
    # Handle plots if it's a string representation of a list
    if isinstance(plots, str):
        if plots.startswith('[') and plots.endswith(']'):
            try:
                import ast
                plots = ast.literal_eval(plots)
            except:
                plots = None
    
    # Handle dataframes if it's a string representation of a dictionary
    if isinstance(dataframes, str):
        if dataframes.startswith('{') and dataframes.endswith('}'):
            try:
                import ast
                dataframes_dict = ast.literal_eval(dataframes)
                # Note: we can't convert string names back to actual DataFrames,
                # so we'll need to handle this at report generation time
                dataframes = dataframes_dict
            except:
                dataframes = None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.{format}"
    report_path = os.path.join(output_dir, report_filename)
    
    try:
        if format == 'html':
            # Generate HTML report
            report_content = _generate_html_report(title, summary, plots, dataframes)
            
            # Write the HTML report to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            message = f"Generated HTML report: {report_path}"
            return report_path, message
        
        elif format == 'pdf':
            # For PDF, first generate HTML content
            html_content = _generate_html_report(title, summary, plots, dataframes)
            
            # Convert to PDF
            try:
                from weasyprint import HTML
                HTML(string=html_content).write_pdf(report_path)
                message = f"Generated PDF report: {report_path}"
                return report_path, message
            except ImportError:
                # Fallback to HTML if WeasyPrint is not available
                html_path = report_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return html_path, f"WeasyPrint not available, generated HTML report instead: {html_path}"
        
        else:
            return None, f"Error: Unsupported report format '{format}'"
    
    except Exception as e:
        return None, f"Error generating report: {str(e)}"

def _generate_html_report(title, summary, plots=None, dataframes=None):
    """
    Generate HTML content for the report.
    
    Args:
        title (str): Report title
        summary (str): Text summary of the analysis
        plots (list, optional): List of plot file paths to include in the report
        dataframes (dict, optional): Dictionary of DataFrames to include in the report, with keys as table names
        
    Returns:
        str: HTML content
    """
    # Initialize Jinja2 environment
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ title }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .section {
                margin-bottom: 30px;
            }
            .plot-container {
                margin: 20px 0;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .summary {
                white-space: pre-line;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #4285f4;
            }
            .footer {
                margin-top: 30px;
                padding-top: 10px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #777;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h1>{{ title }}</h1>
                <p class="summary">{{ summary }}</p>
            </div>
            
            {% if plots %}
            <div class="section">
                <h2>Visualizations</h2>
                {% for plot in plots %}
                <div class="plot-container">
                    <h3>{{ plot.title }}</h3>
                    <img src="data:image/png;base64,{{ plot.data }}" alt="{{ plot.title }}">
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if tables %}
            <div class="section">
                <h2>Data Tables</h2>
                {% for table in tables %}
                <h3>{{ table.title }}</h3>
                {{ table.data }}
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="footer">
                <p>Report generated on {{ timestamp }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    template = jinja2.Template(template_str)
    
    # Prepare plots (convert to base64)
    plot_data = []
    if plots:
        for i, plot_path in enumerate(plots):
            try:
                with open(plot_path, 'rb') as f:
                    plot_base64 = base64.b64encode(f.read()).decode('utf-8')
                    plot_title = f"Plot {i+1}: {os.path.basename(plot_path)}"
                    plot_data.append({'title': plot_title, 'data': plot_base64})
            except Exception as e:
                plot_title = f"Error loading plot {i+1}: {str(e)}"
                plot_data.append({'title': plot_title, 'data': ''})
    
    # Prepare tables
    table_data = []
    if dataframes:
        for table_name, df in dataframes.items():
            if isinstance(df, pd.DataFrame):
                table_html = df.to_html(classes="table table-striped", border=0)
                table_data.append({'title': table_name, 'data': table_html})
            elif isinstance(df, dict):
                # Convert dict to DataFrame for display
                table_html = pd.DataFrame.from_dict(df, orient='index').to_html(classes="table table-striped", border=0)
                table_data.append({'title': table_name, 'data': table_html})
    
    # Render the template
    html_content = template.render(
        title=title,
        summary=summary,
        plots=plot_data,
        tables=table_data,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return html_content 