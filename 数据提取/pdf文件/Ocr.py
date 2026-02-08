
from paddleocr import PaddleOCR


def ocr(image):
    output = "/mnt/f/baidu/"

    # 初始化 PaddleOCR 实例
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    result = ocr.predict(image)

    # 可视化结果并保存 json 结果
    for res in result:
        res.print()
        res.save_to_img(output)
        res.save_to_json(output)




if __name__ == '__main__':
    ocr("/mnt/f/baidu/test.png")
