import os
import json
import logging
from datetime import datetime

class ReActLogger:
    """Logger for the ReAct paradigm, tracking Thoughts, Actions, and Observations."""
    
    def __init__(self, log_dir='./logs', log_to_file=True, log_to_console=True):
        """
        Initialize the ReAct logger.
        
        Args:
            log_dir (str, optional): Directory to save log files
            log_to_file (bool, optional): Whether to log to files
            log_to_console (bool, optional): Whether to log to console
        """
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.current_session_id = None
        self.session_log = []
        
        # Set up standard Python logger
        self.logger = logging.getLogger('react_agent')
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Create log directory if it doesn't exist and we're logging to file
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
    
    def start_session(self, task_description):
        """
        Start a new logging session.
        
        Args:
            task_description (str): Description of the task being performed
            
        Returns:
            str: The session ID
        """
        # Generate a session ID based on timestamp and task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize task summary for filename - replace illegal chars and limit length
        task_summary = "".join(c if c.isalnum() or c == '_' else '_' for c in task_description.replace(" ", "_"))[:20]
        self.current_session_id = f"{timestamp}_{task_summary}"
        
        # Reset session log
        self.session_log = []
        
        # Add task description as first entry
        self.log_task(task_description)
        
        # Set up file handler if logging to file
        if self.log_to_file:
            # Ensure log directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            
            log_file = os.path.join(self.log_dir, f"{self.current_session_id}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Remove any existing file handlers
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Started session {self.current_session_id} for task: {task_description}")
        return self.current_session_id
    
    def log_task(self, task_description):
        """
        Log a task description.
        
        Args:
            task_description (str): Description of the task
        """
        entry = f"Task: {task_description}"
        self.session_log.append(entry)
        self.logger.info(entry)
    
    def log_thought(self, thought):
        """
        Log a thought.
        
        Args:
            thought (str): The thought to log
        """
        entry = f"Thought: {thought}"
        self.session_log.append(entry)
        self.logger.info(entry)
    
    def log_action(self, action, params=None):
        """
        Log an action.
        
        Args:
            action (str): The action name
            params (dict, optional): Action parameters
        """
        if params:
            # Format parameters as a string, handling special types
            params_str_items = []
            for k, v in params.items():
                # Check if the value is a pandas DataFrame and represent it specially
                import inspect
                if v is not None and 'pandas.core.frame.DataFrame' in str(type(v)) or str(v).startswith('<pandas.core.frame.DataFrame'):
                    params_str_items.append(f"{k}=[DataFrame]")
                # Handle dictionaries specially - don't quote them
                elif isinstance(v, dict):
                    params_str_items.append(f"{k}={v}")
                else:
                    params_str_items.append(f"{k}={v!r}")
            
            params_str = ", ".join(params_str_items)
            entry = f"Action: {action}({params_str})"
        else:
            entry = f"Action: {action}"
        
        self.session_log.append(entry)
        self.logger.info(entry)
    
    def log_observation(self, observation):
        """
        Log an observation.
        
        Args:
            observation (str): The observation to log
        """
        entry = f"Observation: {observation}"
        self.session_log.append(entry)
        self.logger.info(entry)
    
    def get_log(self):
        """
        Get the current session log.
        
        Returns:
            list: The current session log entries
        """
        return self.session_log
    
    def save_log_as_json(self, filename=None):
        """
        Save the current session log as a JSON file.
        
        Args:
            filename (str, optional): Custom filename to use
            
        Returns:
            str: Path to the saved JSON file
        """
        if not self.current_session_id:
            return None, "No active session"
        
        if not filename:
            filename = os.path.join(self.log_dir, f"{self.current_session_id}.json")
        
        try:
            # Parse the log entries into structured data
            structured_log = []
            
            current_entry = {}
            for entry in self.session_log:
                if entry.startswith("Task:"):
                    current_entry = {"task": entry[6:].strip()}
                    structured_log.append(current_entry)
                elif entry.startswith("Thought:"):
                    current_entry["thought"] = entry[9:].strip()
                elif entry.startswith("Action:"):
                    current_entry["action"] = entry[8:].strip()
                elif entry.startswith("Observation:"):
                    current_entry["observation"] = entry[13:].strip()
                    # Start a new entry for the next thought-action-observation cycle
                    current_entry = {}
                    structured_log.append(current_entry)
            
            # Remove empty entries
            structured_log = [entry for entry in structured_log if entry]
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(structured_log, f, indent=2)
            
            return filename, f"Log saved to {filename}"
        
        except Exception as e:
            return None, f"Error saving log: {str(e)}" 