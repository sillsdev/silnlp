from silnlp.common.linear_regression import LinearRegressionResult


def test_project_chrf3() -> None:
    result = LinearRegressionResult(version="0.1", slope=50.0, intercept=20.0)

    assert result.project_chrf3(0.8) == 60.0
