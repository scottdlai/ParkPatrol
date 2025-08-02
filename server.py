import cgi
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import numpy as np

from PIL import Image
from ultralytics import YOLO


model = YOLO('model_v2.pt')
PORT = 8000


def predict(img: Image.Image) -> float:
    # img_np = np.array(img)

    results = model(img)

    # results = model.predict(source=img_np, save=True, show_labels=False, show_conf=False, conf=0.60)

    print(f'Results: ${results[0]}')

    probs = [float(box.conf) for box in results[0].boxes]
    occupied_prob = max(probs) if probs else 0.0

    return occupied_prob


class ParkPatrolHandler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != '/api/predict':
            self.send_error(404, "Not found")
            return

        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        if ctype != 'multipart/form-data':
            self.send_error(400, "Only multipart/form-data is supported")
            return

        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(self.headers.get('content-length'))

        fields = cgi.parse_multipart(self.rfile, pdict)
        file_data = fields.get('image')
        if not file_data:
            self.send_error(400, "No file uploaded with field name 'image'")
            return

        try:
            image_bytes = BytesIO(file_data[0])
            img = Image.open(image_bytes)

            print(f'Calling API with img {img}')

            probability = predict(img)

            print(f'Occupied Probability: {probability}')

            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            response = {'occupiedProbability': probability}
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error(500, f"Error processing image: {e}")


if __name__ == '__main__':
    server_address = ('', PORT)
    print(f"Server running on http://localhost:{PORT}")
    httpd = HTTPServer(server_address, ParkPatrolHandler)
    httpd.serve_forever()
