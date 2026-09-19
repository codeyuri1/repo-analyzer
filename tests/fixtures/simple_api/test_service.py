from service import UserService


def test_get_user() -> None:
    assert UserService().get_user()["id"] == "known"
