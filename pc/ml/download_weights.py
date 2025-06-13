# pc/ml/download_weights.py
import os
import requests

def download_weights():
    weights_path = os.path.join(os.path.dirname(__file__), 'yolov3.weights')
    if not os.path.exists(weights_path):
        print("Downloading yolov3.weights...")
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        response = requests.get(url, stream=True)
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
