# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

import os
import sys
import json
import yaml
import shutil
import datetime
import keyboard
from pathlib import Path
from queue import Queue

def load_config(path="config/default.yaml"):
    """
    Load a YAML configuration file.

    Args:
        path (str, optional): Path to the YAML configuration file. Defaults to "config/default.yaml".

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(Path(path), "r") as file:
        return yaml.safe_load(file)


def is_key_pressed(key):
    """
    Check if a specific keyboard key is currently pressed.

    Args:
        key (str): Key name (e.g., 'q', 'space').

    Returns:
        bool: True if the key is pressed, False otherwise.
    """
    return keyboard.is_pressed(key)


def thread_worker(result_queue, func, *args, **kwargs):
    """
    Run a function in a thread and store the result in a queue.

    Args:
        result_queue (Queue): A queue to store the result.
        func (callable): Function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """
    result = func(*args, **kwargs)
    result_queue.put(result)


def streamprinter(text):
    """
    Write text to stdout immediately.

    Args:
        text (str): Text to print.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


def get_next_test_folder(base_dir="./results/tests"):
    """
    Get the next available test folder path.

    Args:
        base_dir (str): Base directory for test folders.

    Returns:
        Path: Path to the next test folder.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("t")]
    numbers = sorted(int(d[1:]) for d in existing if d[1:].isdigit())
    next_id = (numbers[-1] + 1) if numbers else 1
    return base / f"t{next_id:02d}"


def save_test_result(properties: dict, files_to_copy=None, base_dir="./results/tests"):
    """
    Creates a new test result folder, saves the properties.yaml, and copies optional files.

    Args:
        properties (dict): Metadata to save as properties.yaml
        files_to_copy (list of tuples): Each tuple is (src_path, dst_filename)
        base_dir (str): Path to the results/tests directory
    """
    target_folder = get_next_test_folder(base_dir)
    target_folder.mkdir(parents=True)

    # Save properties.yaml
    with open(target_folder / "properties.yaml", "w") as f:
        yaml.dump(properties, f, default_flow_style=False)

    # Copy additional files
    if files_to_copy:
        for src_path, dst_name in files_to_copy:
            shutil.copy(src_path, target_folder / dst_name)

    print(f"[✓] Saved test result to: {target_folder}")
    return str(target_folder)


def saveData(fName, data):
    """
    Save data to a JSON file.

    Args:
        fName (str): File path.
        data (dict or list): Data to save.

    Returns:
        int: 1 if success, 0 otherwise.
    """
    try:
        with open(fName, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return 1
    except:
        return 0


def loadData(fName):
    """
    Load data from a JSON file.

    Args:
        fName (str): File path.

    Returns:
        dict or list: Parsed data.
    """
    data = None
    with open(fName, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def latestDataFile(folder):
    """
    Get the most recent file in a folder.

    Args:
        folder (str): Path to folder.

    Returns:
        str: Path to latest file.
    """
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return folder + str(max(files))


def dataDeNumpyer(data):
    """
    Recursively convert numpy arrays in data to native Python lists.

    Args:
        data (Any): Data to convert.

    Returns:
        Any: Converted data.
    """
    try:
        newData = []
        for datum in data:
            if str(type(datum)) == "<class 'dict'>":
                newData.append(datum)
            elif str(type(datum)) == "<class 'numpy.ndarray'>":
                newData.append(datum.tolist())
            else:
                newData.append(dataDeNumpyer(datum))
        return newData
    except:
        return data


def filenameCreator(folder, filetype):
    """
    Generate a timestamped filename.

    Args:
        folder (str): Folder path.
        filetype (str): File extension or suffix.

    Returns:
        str: Full filename with timestamp.
    """
    dtvar = datetime.datetime.now()
    year = str(dtvar.year)
    month = str(dtvar.month)
    if len(month) == 1:
        month = "0" + month
    day = str(dtvar.day)
    if len(day) == 1:
        day = "0" + day
    hour = str(dtvar.hour)
    if len(hour) == 1:
        hour = "0" + hour
    minute = str(dtvar.minute)
    if len(minute) == 1:
        minute = "0" + minute
    second = str(dtvar.second)
    if len(second) == 1:
        second = "0" + second
    date = year + month + day
    time = hour + minute + second
    return folder + str(date) + "_" + str(time) + filetype
