from app.model import base


class TestOrderByClass:
    def test_orderby_class(self) -> None:
        assert base.OrderBy.STD.name == "STD", "name not match"
        assert (
            base.OrderBy.STD.value == "saletransactiondate"
        ), """
        value not match"""
