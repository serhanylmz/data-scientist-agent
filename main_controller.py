import os
import importlib
import inspect
import sys

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("tools", exist_ok=True)

from llm_interface import LLMInterface, parse_action
from logging_module import ReActLogger

class DataScientistAgent:
    """
    Main controller for the autonomous data scientist agent using the ReAct paradigm.
    """
    
    def __init__(self, api_key=None, model="gpt-4", log_dir="./logs"):
        """
        Initialize the data scientist agent.
        
        Args:
            api_key (str, optional): API key for the LLM provider
            model (str, optional): The model to use
            log_dir (str, optional): Directory for logs
        """
        self.llm = LLMInterface(api_key=api_key, model=model)
        self.logger = ReActLogger(log_dir=log_dir)
        self.available_tools = self._load_available_tools()
        self.current_data = None  # To store the current DataFrame being worked on
        
    def _load_available_tools(self):
        """
        Load all available tools from the tools directory.
        
        Returns:
            dict: Dictionary of available tools
        """
        tools = {}
        
        # Create tools directory if it doesn't exist
        os.makedirs("tools", exist_ok=True)
        
        # Import all modules from the tools directory
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"tools.{module_name}")
                    
                    # Get all functions from the module
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        tools[name] = obj
                except ImportError as e:
                    print(f"Error importing {module_name}: {e}")
        
        return tools
    
    def run(self, task, max_iterations=10):
        """
        Run the data scientist agent on a task.
        
        Args:
            task (str): The task description
            max_iterations (int, optional): Maximum number of iterations
            
        Returns:
            tuple: (final_result, log)
        """
        self.logger.start_session(task)
        
        iterations = 0
        final_result = None
        
        # Dictionary to store results of previous operations
        # This will be used to reference dataframes between function calls
        results_store = {}
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\nIteration {iterations}/{max_iterations}")
            
            # Get the next action and thought from the LLM
            try:
                # Update the prompt to include info about stored results
                stored_data_info = ""
                if self.current_data is not None:
                    stored_data_info = "\nImportant: A DataFrame is already loaded in memory. You can use 'df' directly in your function calls to refer to it."
                
                action_text, thought = self.llm.process_task(task + stored_data_info, self.logger.get_log())
                print(f"Received action text: '{action_text}'")
                print(f"Received thought: '{thought[:50]}...' (truncated)")
                
                # Log the thought
                if thought:
                    self.logger.log_thought(thought)
                else:
                    print("Warning: No thought received from LLM")
                
                # Parse and execute the action
                if not action_text:
                    self.logger.log_observation("No action received from LLM")
                    continue
                    
                func_name, params = parse_action(action_text)
                
                # Check if we should stop (e.g., task complete)
                if func_name == "finish" or func_name == "complete":
                    final_result = params.get("result", "Task completed successfully.")
                    self.logger.log_action("finish", {"result": final_result})
                    break
                
                # Handle special case for 'df' parameter to use current_data
                if 'df' in params and params['df'] == 'df' and self.current_data is not None:
                    # Make a copy to avoid modifying the params dict directly
                    log_params = params.copy()
                    # For logging, replace actual DataFrame with a placeholder
                    log_params['df'] = "[DataFrame]"
                    
                    # Don't quote dictionaries in the params
                    for key, value in log_params.items():
                        if isinstance(value, dict) and key != 'df':
                            log_params[key] = value  # Keep dictionaries as is
                    
                    self.logger.log_action(func_name, log_params)
                    
                    # For execution, use the actual DataFrame
                    params['df'] = self.current_data
                else:
                    # Log the action with original parameters
                    # Don't quote dictionaries in the params
                    log_params = params.copy()
                    for key, value in log_params.items():
                        if isinstance(value, dict):
                            log_params[key] = value  # Keep dictionaries as is
                            
                    self.logger.log_action(func_name, log_params)
                
                # Execute the action if it exists
                if func_name in self.available_tools:
                    try:
                        # Call the tool function
                        result, message = self.available_tools[func_name](**params)
                        
                        # Update current data if the result is a DataFrame
                        import pandas as pd
                        if isinstance(result, pd.DataFrame):
                            self.current_data = result
                            print(f"Stored DataFrame with {len(result)} rows and {len(result.columns)} columns")
                        
                        # Log the observation
                        self.logger.log_observation(message)
                    except Exception as e:
                        # Log the error
                        error_message = f"Error executing {func_name}: {str(e)}"
                        print(f"ERROR: {error_message}")
                        self.logger.log_observation(error_message)
                else:
                    # Log unknown action
                    error_message = f"Unknown action: {func_name}"
                    print(f"ERROR: {error_message}")
                    self.logger.log_observation(error_message)
            except Exception as e:
                error_message = f"Error during iteration {iterations}: {str(e)}"
                print(f"ERROR: {error_message}")
                self.logger.log_observation(error_message)
        
        # Save the final log as JSON
        log_file, _ = self.logger.save_log_as_json()
        
        return final_result, self.logger.get_log()
    
    def execute_action(self, action_name, **params):
        """
        Execute a specific action directly.
        
        Args:
            action_name (str): The name of the action to execute
            **params: Parameters for the action
            
        Returns:
            tuple: (result, message)
        """
        if action_name in self.available_tools:
            try:
                result, message = self.available_tools[action_name](**params)
                return result, message
            except Exception as e:
                return None, f"Error executing {action_name}: {str(e)}"
        else:
            return None, f"Unknown action: {action_name}"

def main():
    """Main function to run the data scientist agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the autonomous data scientist agent.')
    parser.add_argument('task', type=str, help='The task to perform')
    parser.add_argument('--api-key', type=str, help='API key for the LLM provider')
    parser.add_argument('--model', type=str, default='gpt-4', help='The model to use')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum number of iterations')
    
    args = parser.parse_args()
    
    agent = DataScientistAgent(api_key=args.api_key, model=args.model)
    final_result, log = agent.run(args.task, max_iterations=args.max_iterations)
    
    print("\n=== Final Result ===")
    print(final_result)
    
    print("\n=== Summary of Actions ===")
    for entry in log:
        if entry.startswith("Action:"):
            print(entry)

if __name__ == "__main__":
    main() 