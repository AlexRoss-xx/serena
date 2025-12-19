
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "serena", "src"))

from solidlsp.ls_utils import FileUtils

def test_read_file():
    filename = "test_file.txt"
    content = "Line 1\nLine 2\nLine 3"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
        
    try:
        read_content = FileUtils.read_file(filename, "utf-8")
        print(f"Original content length: {len(content)}")
        print(f"Read content length: {len(read_content)}")
        print("--- Read Content ---")
        print(read_content)
        print("--------------------")
        
        if read_content == content:
            print("SUCCESS: Content matches.")
        else:
            print("FAILURE: Content does not match.")
            
    except Exception as e:
        print(f"Error reading file: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_read_file()
