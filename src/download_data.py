import os
import requests
import zipfile
from tqdm import tqdm  

def download_ptbxl(target_folder="./data"):
    # 1. THE CHECK
    csv_path = os.path.join(target_folder, "ptbxl_database.csv")
    if os.path.exists(csv_path):
        print("Dataset already detected in ./data. Skipping download.")
        return

    # 2. SETUP
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_path = os.path.join(target_folder, "ptbxl.zip")
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    # 3. DOWNLOAD 
    print("Downloading PTB-XL dataset (approx. 2.7 GB)...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status() 
        
        total_size = int(r.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Download",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk)) # Aggiorna la barra
                    
        print("\nDownload completed successfully.")
    except Exception as e:
        print(f"\nError during download: {e}")
        return
    
    # 4. EXTRACTION
    print("Extracting files (this may take a few minutes)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
        
    # 5. CLEANUP
    print("Reorganizing directory structure...")
    extracted_folder_name = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    extracted_folder = os.path.join(target_folder, extracted_folder_name)
    
    if os.path.exists(extracted_folder):
        for item in os.listdir(extracted_folder):
            source = os.path.join(extracted_folder, item)
            destination = os.path.join(target_folder, item)
            os.rename(source, destination)
            
        os.rmdir(extracted_folder)
        
    os.remove(zip_path)
    print("Cleanup complete. Dataset is ready in ./data!")

if __name__ == "__main__":
    download_ptbxl()