import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

class LLMInterface:
    """Interface for interacting with Language Model APIs."""
    
    def __init__(self, api_key=None, model="gpt-4"):
        """
        Initialize the LLM interface.
        
        Args:
            api_key (str, optional): API key for the LLM provider. 
                                    If None, attempts to load from OPENAI_API_KEY env variable.
            model (str, optional): The model to use. Defaults to "gpt-4".
        """
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.client = OpenAI(api_key=self.api_key)
    
    def get_response(self, prompt, temperature=0.7, max_tokens=1000):
        """
        Get a response from the LLM.
        
        Args:
            prompt (str): The prompt to send to the LLM
            temperature (float, optional): Controls randomness. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens in response. Defaults to 1000.
            
        Returns:
            str: The LLM's response text
        """
        try:
            system_prompt = """You are a data scientist assistant that follows the ReAct paradigm.
Always structure your responses in two parts:
1. Thought: where you reason about what to do next
2. Action: where you specify an action as a function call, like: function_name(param1=value1, param2=value2)

IMPORTANT DATA HANDLING:
- When a DataFrame is loaded or created, it is automatically stored in memory.
- To reference this DataFrame in subsequent operations, use 'df' (without quotes) as the value for DataFrame parameters.
- Example: After calling read_excel(), you can use: clean_data(df=df, operations={'dropna': True})
- Do NOT use string literals like df="read_excel(...)" which will not work!
- If you want to load data, always use read_excel() first, then in your next action, use 'df' as the parameter value.

IMPORTANT WORKFLOW TIPS:
1. After loading data, always examine the DataFrame structure with examine_dataframe(df=df)
2. Check column names before trying operations on specific columns
3. Use actual DataFrame column names in your operations, not assumed ones
4. Perform operations step by step, checking results after each operation

Available functions:
- read_excel(file_path) - Read data from Excel
- read_sql(query, connection_string) - Read data from SQL
- examine_dataframe(df) - Examine DataFrame structure and columns
- clean_data(df, operations) - Clean the DataFrame
- compute_statistics(df, columns) - Compute summary statistics
- compute_correlations(df, method) - Compute correlation matrix
- generate_plot(df, plot_type, **kwargs) - Generate visualizations
- generate_report(title, summary, plots, dataframes) - Create a report
- finish(result) - Complete the task with a final result

Always end your response with an Action that calls one of these functions.
"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"
    
    def process_task(self, task, log=None):
        """
        Process a task using the ReAct paradigm.
        
        Args:
            task (str): The task description
            log (list, optional): Previous interaction log
            
        Returns:
            tuple: (next_action, thought)
        """
        # Construct prompt with previous log if available
        prompt = f"Task: {task}\n\n"
        
        if log:
            prompt += "Previous steps:\n"
            for entry in log:
                prompt += f"{entry}\n"
            
            # Add a reminder about DataFrame if the last action was a data operation
            df_operations = ["read_excel", "read_sql", "clean_data", "compute_statistics"]
            for op in df_operations:
                if any(f"Action: {op}" in entry for entry in log[-3:]):  # Check recent entries
                    prompt += "\nRemember: A DataFrame is already loaded in memory. Use 'df' directly as a parameter value, not as a string.\n"
                    break
        
        prompt += "\nBased on the above, provide the next step following the ReAct paradigm. Start with 'Thought:' explaining your reasoning, then 'Action:' with a specific action to take."
        
        # Get LLM response
        response = self.get_response(prompt)
        
        # Parse the response to extract Thought and Action
        thought_match = re.search(r"Thought:(.*?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:(.*?)(?=\n\n|$)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return action, thought

def parse_action(action_text):
    """
    Parse the action text from the LLM into a structured format.
    
    Args:
        action_text (str): The action text from the LLM
        
    Returns:
        tuple: (action_name, action_params)
    """
    # Debug: Print raw action text
    print(f"Raw action text: {action_text}")
    
    if not action_text:
        return None, "No action specified"
    
    # Pre-process to handle common DataFrame reference patterns
    action_text = action_text.replace("df=df", "df='df'")
    
    # Pattern to match function call format: function_name(param1=value1, param2=value2, ...)
    pattern = r'(\w+)\s*\((.*)\)'
    match = re.match(pattern, action_text)
    
    if not match:
        return None, f"Could not parse action: {action_text}"
    
    function_name = match.group(1).strip()
    params_text = match.group(2).strip()
    
    # Parse parameters
    params = {}
    if params_text:
        # Handle quoted strings, lists, and other parameter types
        try:
            # Special handling for 'df' parameter to ensure it's not treated as a string
            # This allows 'df=df' to be correctly parsed as using the current DataFrame
            if 'df=df' in params_text or "df='df'" in params_text or 'df="df"' in params_text:
                params['df'] = 'df'
                # Remove this parameter from the text to avoid parsing issues
                patterns = ['df=df', "df='df'", 'df="df"']
                for pattern in patterns:
                    if pattern in params_text:
                        params_text = params_text.replace(pattern, '')
                        # Handle cases where we might have an empty parameter list or trailing comma
                        params_text = params_text.replace('(,', '(').replace(',,', ',').replace(',)', ')')
                        break
            
            # Only parse remaining parameters if there are any
            if params_text and not params_text.isspace() and params_text != ',':
                # Convert to valid JSON for proper parsing
                # Replace Python syntax with JSON syntax
                json_params = "{" + params_text.replace("=", ":").replace("'", "\"") + "}"
                try:
                    parsed_params = json.loads(json_params)
                    # Add these to our parameters
                    for k, v in parsed_params.items():
                        params[k] = v
                except:
                    # If JSON parsing fails, fall back to regex-based parsing
                    param_pairs = params_text.split(',')
                    for pair in param_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Special handling for df parameter
                            if key == 'df' and (value == 'df' or value == "'df'" or value == '"df"'):
                                params[key] = 'df'
                                continue
                            
                            # Try to convert string values to appropriate types
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif value.isdigit():
                                value = int(value)
                            elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                                value = float(value)
                            elif value.startswith('[') and value.endswith(']'):
                                # Simple list parsing
                                value = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                            elif value.startswith('{') and value.endswith('}'):
                                # Simple dict parsing
                                try:
                                    value = json.loads(value.replace("'", "\""))
                                except:
                                    pass  # Keep as string if parsing fails
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            elif value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                                
                            params[key] = value
                
        except Exception as e:
            print(f"Error parsing parameters: {e}")
            # Even if parsing fails, ensure 'df' parameter is handled if present
            if 'df=df' in params_text or "df='df'" in params_text or 'df="df"' in params_text:
                params['df'] = 'df'
    
    # Make sure we didn't miss the df reference
    if 'df=df' in action_text or "df='df'" in action_text or 'df="df"' in action_text:
        params['df'] = 'df'
        
    return function_name, params 