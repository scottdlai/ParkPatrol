import cgi
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import base64
import json

from PIL import Image, ImageDraw
from ultralytics import YOLO


model = YOLO('models_trained/modelv8m.pt')
PORT = 8000
UNOCCUPIED_CLASS_ID = 1

def predict(img: Image.Image) -> tuple[float, Image.Image]:
    results = model.predict(
        source=img,
        conf=0.60,
        show=False,
        save=False,
        show_labels=False,
        show_conf=False,
        verbose=False
    )

    boxes = results[0].boxes
    probs = [float(box.conf) for box in boxes]
    occupied_prob = sum(probs) / len(probs) if probs else 0.0

    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id == UNOCCUPIED_CLASS_ID:
            xyxy = box.xyxy[0].tolist()
            draw.rectangle(xyxy, outline="green", width=3)
        else:
            xyxy = box.xyxy[0].tolist()
            draw.rectangle(xyxy, outline="red", width=3)

    return occupied_prob, annotated_img


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

            (probability, annotated_img) = predict(img)

            print(f'Occupied Probability: {probability}')

            # Convert image to base64-encoded PNG
            buf = BytesIO()
            annotated_img.save(buf, format='PNG')
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{img_base64}"

            response = {
                'occupiedProbability': probability,
                'img': data_url
            }

            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_error(500, f"Error processing image: {e}")


if __name__ == '__main__':
    server_address = ('', PORT)
    print(f"Server running on http://localhost:{PORT}")
    httpd = HTTPServer(server_address, ParkPatrolHandler)
    httpd.serve_forever()
