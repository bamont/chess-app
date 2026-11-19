import sys

MOVES = ["e2e4", "d2d4", "g1f3", "c2c4"]
FINAL_LINES = 3


def main() -> None:
    lines = 1
    for raw in sys.stdin:
        command = raw.split()
        if not command:
            continue
        if command[0] == "uci":
            print("id name FakeMultiPv", flush=True)
            print("option name MultiPV type spin default 1 min 1 max 500", flush=True)
            print("uciok", flush=True)
        elif command[0] == "isready":
            print("readyok", flush=True)
        elif command[0] == "setoption" and "MultiPV" in command:
            lines = int(command[command.index("value") + 1])
        elif command[0] == "go":
            for index in range(1, min(lines, len(MOVES)) + 1):
                print(f"info depth 2 multipv {index} score cp {60 - 10 * index} pv {MOVES[index - 1]}", flush=True)
            for index in range(1, min(lines, FINAL_LINES) + 1):
                print(
                    f"info depth 3 multipv {index} score cp {50 - 10 * index} pv {MOVES[index - 1]} e7e5",
                    flush=True,
                )
            print(f"bestmove {MOVES[0]}", flush=True)
        elif command[0] == "quit":
            break


if __name__ == "__main__":
    main()