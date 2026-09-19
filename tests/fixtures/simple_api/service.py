from repository import UserRepository


class UserService:
    def get_user(self) -> dict[str, str]:
        return UserRepository().find()
