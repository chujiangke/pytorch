import requests as req


def demo(url, files, data=None):
    result = req.post(url, data=data, files=files).text
    return result


if __name__ == "__main__":
    url = "http://127.0.0.1:5000/image_classification"
    files = {"image": open("img/workflow.jpg", "rb")}
    data = {"delete_file": True}
    r = demo(url, files, data)
    print(r)

