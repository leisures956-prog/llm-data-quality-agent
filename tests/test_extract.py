from app.pipeline.extract import extract_document


def test_extract_main_text():
    html = """
    <html>
        <head><title>测试网页</title></head>
        <body>
            <nav>广告 首页 登录</nav>
            <article>
                <p>这是正文内容。</p>
            </article>
        </body>
    </html>
    """

    result = extract_document(html)

    assert result["title"] == "测试网页"
    assert "这是正文内容" in result["clean_text"]
    assert "广告" not in result["clean_text"]