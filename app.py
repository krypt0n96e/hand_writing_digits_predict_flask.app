# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import transforms
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
# from PIL import Image
# from module import Model
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# # Load the state dictionary of the model from the model.dth file
# # model_state = torch.load('model.dth', map_location=torch.device('cpu'))
# model_state = torch.load('my_cp.tar', map_location=torch.device('cpu'))

# # Create an instance of your model
# model = Model()

# # Load the state dictionary into the model
# model.load_state_dict(model_state['state_dict'])

# # Set the model to evaluation mode
# model.eval()


# # # Define transformations to apply to the input image
# # preprocess = transforms.Compose([
# #     transforms.Grayscale(num_output_channels=1),
# #     transforms.Resize((28, 28)),
# #     transforms.ToTensor(),
# # ])

# # Định nghĩa các phép biến đổi tiền xử lý
# preprocess = transforms.Compose([
#     transforms.Resize((28, 28)),  # Chuyển đổi kích thước ảnh về (28, 28)
#     # transforms.ColorJitter(contrast=2),  # Tăng độ tương phản
#     transforms.Grayscale(),        # Chuyển ảnh sang ảnh xám
#     F.invert,                      # Đảo ngược màu
#     transforms.ToTensor(),         # Chuyển ảnh sang tensor
#     transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa ảnh về khoảng [-1, 1]
# ])

# # Define the function to predict the digit
# def predict_digit(image):
#     processed_image = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(processed_image)
#     _, predicted = torch.max(output, 1)

#     # Chuyển đổi tensor thành mảng numpy và loại bỏ chiều thứ nhất
#     processed_image = processed_image.squeeze().numpy()
    
#     # In ra ảnh đã tiền xử lý
#     plt.imshow(processed_image, cmap='gray')
#     plt.axis('off')
#     plt.show()
#     return predicted.item()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     image = Image.open(file.stream).convert('L')  # Convert to grayscale
#     digit = predict_digit(image)
    
#     return jsonify({'digit': digit})

# if __name__ == '__main__':
#     app.run(debug=True,host="0.0.0.0")


from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from module import Model
import os
from io import BytesIO
import base64

app = Flask(__name__)

processed_image_path = 'static/processed_image.jpg'

# Load the state dictionary of the model from the model.dth file
# model_state = torch.load('my_cp.tar', map_location=torch.device('cpu'))
model_state = torch.load('model.dth', map_location=torch.device('cpu'))

# Create an instance of your model
model = Model()

# Load the state dictionary into the model
# model.load_state_dict(model_state['state_dict'])
model.load_state_dict(model_state)

# Set the model to evaluation mode
model.eval()

# Define the transformations to preprocess the input image
preprocess = transforms.Compose([
    # transforms.Grayscale(),                 # Convert the image to grayscale
    # transforms.Resize((28, 28)),            # Resize the image to (28, 28)
    # transforms.ColorJitter(contrast=1.5),     # Increase contrast
    # F.invert,                      # Đảo ngược màu
    transforms.ToTensor(),                  # Convert the image to tensor
    transforms.Normalize((0.5,), (0.5,))    # Normalize the image
])

# Define the function to predict the digit
def predict_digit(image):
    processed_image = preprocess(image)
    # Convert the processed image tensor back to PIL Image
    processed_image_pil = transforms.ToPILImage()(processed_image)
    # Save the processed image to disk
    processed_image_pil.save(processed_image_path)
    processed_image = processed_image.unsqueeze(0)
    with torch.no_grad():
        output = model(processed_image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file.stream)
    
    image = image.rotate(0, expand=True)

    # Convert to grayscale
    image_gray = image.convert("L")

    # Invert colors (dark text on light background to light text on dark background)
    image_inverted = Image.eval(image_gray, lambda x: 255 - x)

    # Resize the image to (28, 28)
    image_resized = image_inverted.resize((28, 28))

    digit = predict_digit(image_resized)
    
    return jsonify({'digit': digit, 'processed_image_path': processed_image_path})
@app.route('/upload_drawing', methods=['POST'])
def upload_drawing():
    # Nhận dữ liệu hình ảnh từ canvas
    image_data = request.form['imageData']

    # Tách phần dữ liệu base64 từ URL dữ liệu hình ảnh
    _, encoded_data = image_data.split(',')

    # Giải mã dữ liệu base64 thành dữ liệu nhị phân
    decoded_data = base64.b64decode(encoded_data)

    # Chuyển dữ liệu nhị phân thành đối tượng hình ảnh PIL
    image = Image.open(BytesIO(decoded_data))

    # Lưu ảnh đã giải mã thành tệp tin upload.png
    upload_path = os.path.join('static', 'upload.png')
    image.save(upload_path)

    # Chuyển đổi sang ảnh xám
    image_gray = image.convert("L")

    # Đảo ngược màu (chữ đen trên nền sáng thành chữ sáng trên nền đen)
    image_inverted = Image.eval(image_gray, lambda x: 255 - x)

    # Thay đổi kích thước ảnh thành (28, 28)
    image_resized = image_inverted.resize((28, 28))

    # Thực hiện dự đoán bằng model và lưu ảnh đã xử lý
    digit = predict_digit(image_resized)
    
    return jsonify({'digit': digit})
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
