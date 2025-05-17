import os
import streamlit as st
from config import DATA_DIR, SUPPORTED_FILE_EXTENSIONS 
import logging

logger = logging.getLogger(__name__)

def ensure_data_directory():
    """Ensures the DATA_DIR exists, creating it if necessary."""
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            st.info(f"Created data directory: {DATA_DIR}. Please add your documents here for RAG.")
            logger.info(f"Created data directory: {DATA_DIR}")
        except OSError as e:
            st.error(f"Failed to create data directory {DATA_DIR}: {e}")
            logger.error(f"Failed to create data directory {DATA_DIR}: {e}", exc_info=True)
            # Depending on the app's needs, you might want to stop execution here
            # or handle it in a way that the app can still run in a limited mode.
            # For now, we'll let Streamlit continue and subsequent operations might fail.

def save_uploaded_file(uploaded_file_object) -> bool:
    """Saves an uploaded file to the DATA_DIR if it's a supported type."""
    if not uploaded_file_object:
        return False

    file_name = uploaded_file_object.name
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext not in SUPPORTED_FILE_EXTENSIONS:
        st.error(f"Unsupported file type: '{file_ext}'. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
        logger.warning(f"Attempted to upload unsupported file type: {file_name}")
        return False

    file_path = os.path.join(DATA_DIR, file_name)

    if os.path.exists(file_path):
        logger.info(f"File '{file_name}' already exists in {DATA_DIR}. It will be overwritten.")
        # Consider adding a user confirmation here for overwriting in a real application

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file_object.getbuffer())
        logger.info(f"Successfully saved uploaded file: {file_name} to {DATA_DIR}")
        return True
    except IOError as e:
        st.error(f"Error saving file '{file_name}': {e}")
        logger.error(f"Could not save uploaded file {file_name}: {e}", exc_info=True)
        return False
    except Exception as e: # Catch any other potential errors during file saving
        st.error(f"An unexpected error occurred while saving '{file_name}': {e}")
        logger.error(f"Unexpected error saving uploaded file {file_name}: {e}", exc_info=True)
        return False


def delete_file_from_data_dir(file_name: str) -> bool:
    """
    Deletes a file from the DATA_DIR.
    
    Args:
        file_name: Name of the file to delete.
    
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Successfully deleted file: {file_name} from {DATA_DIR}")
            return True
        except OSError as e:
            st.error(f"Error deleting file '{file_name}': {e}")
            logger.error(f"Could not delete file {file_name}: {e}", exc_info=True)
            return False
        except Exception as e:
            st.error(f"An unexpected error occurred while deleting '{file_name}': {e}")
            logger.error(f"Unexpected error deleting file {file_name}: {e}", exc_info=True)
            return False
    else:
        st.warning(f"File '{file_name}' not found in {DATA_DIR} or is not a file.")
        logger.warning(f"Attempted to delete non-existent or non-file item: {file_name}")
        return False

def list_files_in_data_dir() -> list:
    """
    Lists supported files in the DATA_DIR.
    
    Returns:
        list: List of dictionaries containing file information (name, size, path).
    """
    files_info = []
    if not os.path.exists(DATA_DIR) or not os.path.isdir(DATA_DIR):
        logger.warning(f"Data directory '{DATA_DIR}' does not exist or is not a directory.")
        return files_info

    try:
        for f_name in os.listdir(DATA_DIR):
            f_path = os.path.join(DATA_DIR, f_name)
            if os.path.isfile(f_path) and os.path.splitext(f_name)[1].lower() in SUPPORTED_FILE_EXTENSIONS:
                try:
                    files_info.append({
                        "name": f_name,
                        "size": os.path.getsize(f_path),
                        "path": f_path
                    })
                except OSError as e:
                    logger.warning(f"Could not get info for file: {f_path}. Error: {e}")
    except OSError as e:
        logger.error(f"Could not list files in data directory '{DATA_DIR}': {e}", exc_info=True)
        st.error(f"Error accessing data directory: {e}")
    return files_info