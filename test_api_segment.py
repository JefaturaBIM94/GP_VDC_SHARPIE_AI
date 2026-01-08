# test_api_segment.py
import requests

API_URL = "http://127.0.0.1:8000/api/segment"

image_path = "test_image.jpeg"  # ajusta si tu archivo tiene otro nombre
prompt = "columns"
threshold = 0.5

files = {
    "file": open(image_path, "rb"),
}
data = {
    "prompt": prompt,
    "threshold": str(threshold),
    "session_id": "",
    "force_new_session": "true",
}

resp = requests.post(API_URL, files=files, data=data)
print("Status code:", resp.status_code)
print("JSON keys:", resp.json().keys())
print("Summary:\n", resp.json()["summary"])
print("Num objects:", len(resp.json()["objects"]))
