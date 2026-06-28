# validate_split.py
import json

def validate_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"File: {path} | Total samples: {len(lines)}")
    
    for idx, line in enumerate(lines):
        try:
            data = json.loads(line)
            assert "messages" in data, f"Line {idx}: Missing messages key"
            msgs = data["messages"]
            assert len(msgs) == 3, f"Line {idx}: Must have exactly 3 role messages"
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"
            assert len(msgs[2]["content"]) > 0, f"Line {idx}: Empty assistant output"
        except Exception as e:
            print(f"Validation FAILED on line {idx}: {e}")
            return False
            
    print(f"Validation PASSED for {path}")
    return True

if __name__ == "__main__":
    v1 = validate_file("./datapack/train.jsonl")
    v2 = validate_file("./datapack/val.jsonl")
    if v1 and v2:
        print("\nReady for Sprint 2 Fine-Tuning Environment Setup.")