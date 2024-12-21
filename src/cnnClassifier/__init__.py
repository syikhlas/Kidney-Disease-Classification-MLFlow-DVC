import os
import sys
import logging

# Logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Ensure logs directory exists
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level
    format=logging_str,  # Format for log messages
    handlers=[
        logging.FileHandler(log_filepath),  # Log to file
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

# Define logger
logger = logging.getLogger("cnnClassifierLogger")
