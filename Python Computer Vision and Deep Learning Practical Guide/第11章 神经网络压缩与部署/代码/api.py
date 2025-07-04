# 使用flask为深度学习模型创建访问接口
# 以实现与不同语言项目的对接

from flask import Flask, jsonify, request
import logging
from werkzeug.utils import secure_filename
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch
import os
from time import ctime

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "tmp/img"
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg", "gif"])

transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

net = resnet18()

def recognition(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)
    result = net(img_tensor)
    label = torch.argmax(result, dim=1)
    return label

def allowed_file(filename):
    # 判断图片名称是否符合要求
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/image_classification", methods=["POST"])
def run(delete_file=True):
    img = request.files["image"]
    if img and allowed_file(img.filename):
        # 保存上传来的图片
        filename = secure_filename(img.filename)
        folder = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])
        img_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img.save(img_path)
    else:
        app.logger.error("Image not available .")
    label = recognition(img_path)
    app.logger.info("Result : {}".format(str(label)))
    if delete_file:
        os.remove(img_path)
    return str(label)


@app.before_request
def before_request():
    ip = request.remote_addr
    app.logger.info("Time : {} Remote ip : {}".format(ctime(), ip))


if __name__ == "__main__":
    # app.debug = True
    handler = logging.FileHandler("flask.log")
    app.logger.addHandler(handler)
    # logger默认只在debug模式下纪录，但是部署不可能用debug模式
    # 所以要纪录日志的话，要先把日志级别设置为debug级别
    app.logger.setLevel(logging.DEBUG)
    app.run(host="127.0.0.1", port=5000)
