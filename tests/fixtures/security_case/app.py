import subprocess


def run_user_command(command: str) -> None:
    subprocess.run(command, shell=True, check=False)


def safe_value(value: str) -> str:
    return value.strip()
