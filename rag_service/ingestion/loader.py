'''Loader placeholder'''
from pathlib import Path

def scan_documents(base_dir = "soucepdf"):
    docs = []
    base =Path(base_dir)

    if not base.exists():
        raise FileNotFoundError(f"The directory {base_dir} does not exist.")
    
    for file_path in base.rglob("*"):
        if file_path.suffix.lower() in [".pdf"]:
            parts = file_path.relative_to(base).parts
            if len(parts) >= 3:
                company, policy_type, file_name = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                company, policy_type = parts[0], "unknown"
                file_name = parts[1]
            else:
                company, policy_type = "unknown", "unknown"
                file_name = parts[0]
         
            docs.append({
                "path": str(file_path),
                "company": company,
                "policy_type": policy_type,
                "file_name": file_name
            })
    print(f"Scanned {len(docs)} documents from {base_dir}")
    return docs
