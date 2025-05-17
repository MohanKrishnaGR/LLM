"""
Document tracking utilities for managing file states and changes.
"""
import os
import hashlib
import json
import logging # Recommended to use logging over print for consistency
from config import SUPPORTED_FILE_EXTENSIONS # New import

logger = logging.getLogger(__name__)

def get_file_hash(file_path: str) -> str:
    """Computes the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192): # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"Could not read file {file_path} for hashing: {e}")
        raise # Re-raise the exception to be handled by the caller

def scan_directory_for_document_states(directory: str) -> dict:
    """
    Scans a directory for supported files and returns their current states.
    
    Returns:
        dict: A dictionary containing file paths as keys and their states (hash, mtime) as values.
    """
    doc_states = {}
    if not os.path.isdir(directory):
        logger.error(f"Data directory '{directory}' not found. Cannot scan for documents.")
        return doc_states # Return empty if directory doesn't exist

    logger.info(f"Scanning directory '{directory}' for supported file types: {SUPPORTED_FILE_EXTENSIONS}")
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_ext = os.path.splitext(file_name)[1].lower() # Get extension and lowercase it

            if file_ext in SUPPORTED_FILE_EXTENSIONS:
                try:
                    file_hash = get_file_hash(file_path)
                    doc_states[file_path] = {
                        "hash": file_hash,
                        "last_modified": os.path.getmtime(file_path), # Store last modified time
                        "file_name": file_name # Store filename for easier reference if needed
                    }
                    logger.debug(f"Tracked supported file: {file_path}")
                except Exception as e:
                    # Log error if hashing or stat-ing fails for a supported file type
                    logger.error(f"Error processing file {file_path} during scan: {e}", exc_info=True)
            else:
                logger.debug(f"Skipping unsupported file type: {file_path} (extension: {file_ext})")
    
    if not doc_states:
        logger.warning(f"No files with supported extensions found in directory: {directory}. Ensure files have extensions: {SUPPORTED_FILE_EXTENSIONS}")
    else:
        logger.info(f"Found {len(doc_states)} supported files in '{directory}'.")
    return doc_states

def load_tracking_data(file_path: str) -> dict:
    """Loads the document tracking data from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                logger.info(f"Successfully loaded tracking data from {file_path}")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading tracking data from {file_path}: {e}. Returning empty data.", exc_info=True)
            return {} # Return empty dict on error to allow rebuilding
    logger.info(f"Tracking file {file_path} not found. Returning empty data.")
    return {}

def save_tracking_data(file_path: str, data: dict):
    """Saves the document tracking data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved tracking data for {len(data)} documents to {file_path}")
    except IOError as e:
        logger.error(f"Error saving tracking data to {file_path}: {e}", exc_info=True)


def compare_document_states(current_states: dict, tracked_states: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Compares current document states (from filesystem scan) with previously tracked states.
    Returns lists of new file paths, modified file paths, and deleted document IDs (which are file paths).
    """
    new_files: list[str] = []
    modified_files: list[str] = []

    current_paths = set(current_states.keys())
    tracked_paths = set(tracked_states.keys())

    # Identify new files: present in current_paths but not in tracked_paths
    for path in current_paths - tracked_paths:
        new_files.append(path)
        logger.info(f"Detected new file: {current_states[path].get('file_name', path)}")

    # Identify potentially modified files: present in both current_paths and tracked_paths
    for path in current_paths.intersection(tracked_paths):
        # Compare hashes to confirm modification
        if current_states[path]["hash"] != tracked_states[path].get("hash"): # Use .get for safety
            modified_files.append(path)
            logger.info(f"Detected modified file: {current_states[path].get('file_name', path)} (hash changed)")

    # Identify deleted documents: present in tracked_paths but not in current_paths
    # The doc_id used for deletion from the index is the file_path.
    deleted_doc_ids: list[str] = list(tracked_paths - current_paths)
    if deleted_doc_ids:
        deleted_file_names = [tracked_states[path].get('file_name', path) for path in deleted_doc_ids]
        logger.info(f"Detected deleted files (doc_ids/paths): {deleted_file_names}")

    if not new_files and not modified_files and not deleted_doc_ids:
        logger.info("No changes detected in document states.")
        
    return new_files, modified_files, deleted_doc_ids