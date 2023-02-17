from yolo_unipose.yolo_unipose import yolo_unipose

def test_yolo():
    model = yolo_unipose()
    img = './tests/data/rxy.jpg'
    bbox, class_name, confidence = model(img)
    assert len(bbox) == 4
    assert type(class_name) == str
    assert confidence > 0.8