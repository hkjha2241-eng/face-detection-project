import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    
    file.save(input_path)
    
    results = model(input_path, conf=0.6, agnostic_nms=True)
    results[0].save(filename=output_path)
    
    return jsonify({'result_image': '/static/uploads/output.jpg'})

if __name__ == '__main__':
    app.run(debug=True)