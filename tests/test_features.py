from app.pipeline.features import compute_features


def test_compute_basic_features():
    html = """
    <html>
        <head>
            <title>网页数据质量</title>
        </head>
        <body>
            <nav>首页 广告 登录</nav>
            <article>
                <p>这是一段正文内容。</p>
                <p>这是第二段正文内容。</p>
            </article>
        </body>
    </html>
    """

    features = compute_features(
        html=html,
        title="网页数据质量",
        clean_text="这是一段正文内容。\n\n这是第二段正文内容。",
    )

    assert features["raw_text_chars"] > 0
    assert features["clean_text_chars"] > 0
    assert features["paragraph_count"] == 2
    assert 0 <= features["link_ratio"] <= 1
    assert 0 <= features["removed_text_ratio"] <= 1


def test_repeated_lines_are_detected():
    html = "<html><body><p>测试</p></body></html>"

    features = compute_features(
        html=html,
        title="测试",
        clean_text="重复内容\n重复内容\n普通内容",
    )

    assert features["repeated_line_ratio"] > 0