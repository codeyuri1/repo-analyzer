from service import UserService


def main() -> None:
    print(UserService().get_user())


if __name__ == "__main__":
    main()
