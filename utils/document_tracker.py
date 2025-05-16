# LLM/utils/document_tracker.py
import os
import hashlib
import json
from datetime import datetime # Corrected: Added space here
import logging

logger = logging.getLogger(__name__)

def get_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192): # Use walrus operator for cleaner loop
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found when trying to hash: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}", exc_info=True)
        return None

def get_file_m_time(filepath):
    """Gets the last modification time of a file as an ISO format string."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
    except FileNotFoundError:
        logger.error(f"File not found when trying to get m_time: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error getting m_time for file {filepath}: {e}", exc_info=True)
        return None

def scan_directory_for_document_states(data_dir: str) -> dict:
    """
    Scans a directory and returns a dictionary of document states (hash, m_time).
    Keys are absolute file paths.
    """
    doc_states = {}
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory {data_dir} not found for scanning.")
        return doc_states
        
    try:
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                # Add more sophisticated file type filtering here if needed
                # e.g., based on a list of supported extensions
                # supported_extensions = ('.txt', '.pdf', '.md', '.docx', '.pptx')
                # if not filename.lower().endswith(supported_extensions):
                #     logger.debug(f"Skipping unsupported file type: {filepath}")
                #     continue

                file_hash = get_file_hash(filepath)
                m_time = get_file_m_time(filepath)
                
                if file_hash and m_time:
                    # Using absolute filepath as doc_id for uniqueness
                    doc_states[filepath] = {
                        "hash": file_hash,
                        "m_time": m_time,
                        "filename": filename # Storing filename for convenience
                    }
                else:
                    logger.warning(f"Could not get hash or m_time for {filepath}. Skipping.")
    except Exception as e:
        logger.error(f"Error scanning directory {data_dir}: {e}", exc_info=True)
    return doc_states

def load_tracking_data(tracking_file_path: str) -> dict:
    """Loads document tracking data from a JSON file."""
    if os.path.exists(tracking_file_path):
        try:
            with open(tracking_file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(f"Tracking data in {tracking_file_path} is not a dictionary. Returning empty.")
                    return {}
                return data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {tracking_file_path}. Returning empty tracking data.", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Error loading tracking data from {tracking_file_path}: {e}", exc_info=True)
            return {}
    return {}

def save_tracking_data(tracking_file_path: str, doc_states: dict):
    """Saves document tracking data to a JSON file."""
    try:
        # Ensure the directory for the tracking file exists
        os.makedirs(os.path.dirname(tracking_file_path), exist_ok=True)
        with open(tracking_file_path, 'w') as f:
            json.dump(doc_states, f, indent=4)
        logger.info(f"Document tracking data saved to {tracking_file_path}")
    except Exception as e:
        logger.error(f"Error saving tracking data to {tracking_file_path}: {e}", exc_info=True)

def compare_document_states(current_states: dict, tracked_states: dict) -> tuple[list, list, list]:
    """
    Compares current document states with tracked states to find new, modified, and deleted files.
    
    Args:
        current_states: Dictionary of current document states (from scanning the filesystem).
                        Keys are file paths.
        tracked_states: Dictionary of previously tracked document states (from the tracking file).
                        Keys are file paths.

    Returns:
        tuple: (new_file_paths, modified_file_paths, deleted_doc_ids)
               Paths/IDs are typically the file paths used as keys in the state dictionaries.
    """
    new_file_paths = []
    modified_file_paths = []
    
    current_doc_ids = set(current_states.keys())
    tracked_doc_ids = set(tracked_states.keys())

    # New files: in current_states but not in tracked_states
    for doc_id in current_doc_ids - tracked_doc_ids:
        new_file_paths.append(doc_id)
        logger.debug(f"Detected new file: {doc_id}")

    # Deleted files: in tracked_states but not in current_states
    deleted_doc_ids = list(tracked_doc_ids - current_doc_ids)
    if deleted_doc_ids:
        logger.debug(f"Detected deleted files/IDs: {deleted_doc_ids}")


    # Potentially modified files: in both, check hash or m_time
    for doc_id in current_doc_ids.intersection(tracked_doc_ids):
        current_hash = current_states[doc_id].get("hash")
        tracked_hash = tracked_states[doc_id].get("hash")
        current_m_time = current_states[doc_id].get("m_time")
        tracked_m_time = tracked_states[doc_id].get("m_time")

        # Primary check: hash difference
        if current_hash != tracked_hash:
            modified_file_paths.append(doc_id)
            logger.debug(f"Detected modified file (hash mismatch): {doc_id}")
        # Secondary check: m_time difference (if hashes are same, or one is missing)
        # This handles cases where content might not change hash (e.g. metadata only)
        # or if hashing failed for one version.
        elif current_m_time != tracked_m_time:
            modified_file_paths.append(doc_id)
            logger.debug(f"Detected modified file (m_time mismatch, hash was {current_hash==tracked_hash}): {doc_id}")
            
    return new_file_paths, modified_file_paths, deleted_doc_ids
