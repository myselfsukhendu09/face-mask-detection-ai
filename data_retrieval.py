
import os
import requests

def download_sample_images():
    samples = {
        "masked_1.jpg": "https://raw.githubusercontent.com/prajnasb/observations/master/mask_classifier/Data_set/with_mask/with_mask_1.jpg",
        "unmasked_1.jpg": "https://raw.githubusercontent.com/prajnasb/observations/master/mask_classifier/Data_set/without_mask/without_mask_1.jpg"
    }
    
    os.makedirs('data/samples', exist_ok=True)
    
    for name, url in samples.items():
        print(f"Downloading {name}...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"data/samples/{name}", 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {name}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    download_sample_images()
