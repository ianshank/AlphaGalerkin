
import torch
import sys

def inspect(path):
    print(f"Loading {path}")
    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            print("Keys:", data.keys())
        else:
            print("Not a dict, type:", type(data))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    inspect(sys.argv[1])
