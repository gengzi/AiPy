from paddleocr import PPStructureV3


def pdfOcr(pdfPath, outputPath):
    # 初始化 PP-StructureV3
    table_engine = PPStructureV3(
        lang='ch'
    )

    # 直接处理 PDF 文件
    result = table_engine.predict(input=pdfPath)  # 直接传入 PDF 路径即可

    # 保存结果（可选，支持 JSON/Excel 等格式）
    # save_structure_res(result, outputPath, pdf_path)

    # 打印解析结果
    for res  in result:
        print(res)
        res.save_to_json(outputPath)
        res.save_to_img(outputPath)
        res.save_to_markdown(outputPath)





if __name__ == '__main__':
    pdfOcr( "/mnt/f/baidu/RAG三问【耗时整理 免费分享 cunlove.cn】.pdf","/mnt/f/baidu/")