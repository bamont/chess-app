import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess.board import Board
from chess.notation import move_to_uci, parse_uci
from chess.rules import generate_legal_moves


def main() -> None:
    fen: str | None = None
    moves: list[str] = []
    lines = 1
    for line in sys.stdin:
        command = line.split()
        if not command:
            continue
        if command[0] == "uci":
            print("id name FakeEngine", flush=True)
            print("option name UCI_LimitStrength type check default false", flush=True)
            print("option name UCI_Elo type spin default 1320 min 1320 max 3190", flush=True)
            print("option name MultiPV type spin default 1 min 1 max 500", flush=True)
            print("uciok", flush=True)
        elif command[0] == "isready":
            print("readyok", flush=True)
        elif command[0] == "setoption" and "MultiPV" in command:
            lines = int(command[command.index("value") + 1])
        elif command[0] == "position":
            fen = " ".join(command[2:8]) if command[1] == "fen" else None
            moves = command[command.index("moves") + 1 :] if "moves" in command else []
        elif command[0] == "go":
            board = Board.from_fen(fen) if fen is not None else Board.starting_position()
            for text in moves:
                board.make_move(parse_uci(text, generate_legal_moves(board, board.turn)))
            legal = generate_legal_moves(board, board.turn)
            if legal:
                for index, move in enumerate(legal[:lines], start=1):
                    print(f"info depth 5 multipv {index} score cp {40 - 10 * index} pv {move_to_uci(move)}", flush=True)
                print(f"bestmove {move_to_uci(legal[0])}", flush=True)
            else:
                print("info depth 0 score mate 0", flush=True)
                print("bestmove (none)", flush=True)
        elif command[0] == "quit":
            break


if __name__ == "__main__":
    main()