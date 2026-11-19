import sys


def main() -> None:
    for raw in sys.stdin:
        command = raw.split()
        if not command:
            continue
        if command[0] == "uci":
            print("id name FakePlain", flush=True)
            print("uciok", flush=True)
        elif command[0] == "isready":
            print("readyok", flush=True)
        elif command[0] == "go":
            print("info depth 4 score cp 25 pv e2e4 e7e5", flush=True)
            print("bestmove e2e4", flush=True)
        elif command[0] == "quit":
            break


if __name__ == "__main__":
    main()