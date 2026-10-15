"""A tiny UCI engine for the label/distill tests (no engine binary needed).

Scores every legal move by the material balance after it (one ply), with a
mate in 1 reported as "score mate 1", and answers MultiPV like a real
engine: one "info ... multipv i score ... pv <move>" line per requested
line, best first, then "bestmove". Run as: python fake_uci_engine.py
"""

import sys

import chess

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color)


def score_after(board: chess.Board, move: chess.Move):
    mover = board.turn
    board.push(move)
    try:
        if board.is_checkmate():
            return "mate", 1
        if board.is_stalemate() or board.is_insufficient_material():
            return "cp", 0
        return "cp", material(board, mover) - material(board, not mover)
    finally:
        board.pop()


def sort_key(item):
    (kind, value), move = item
    return (1, -value, move.uci()) if kind == "mate" else (0, value, move.uci())


def main() -> None:
    board = chess.Board()
    multipv = 1
    out = sys.stdout
    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            out.write("id name FakeEngine\nid author NAGS tests\n"
                      "option name MultiPV type spin default 1 min 1 max 256\n"
                      "option name Hash type spin default 16 min 1 max 1024\n"
                      "option name Threads type spin default 1 min 1 max 1\n"
                      "option name Clear Hash type button\n"
                      "uciok\n")
        elif cmd == "isready":
            out.write("readyok\n")
        elif cmd == "setoption" and "name" in tokens:
            rest = tokens[tokens.index("name") + 1:]
            name = " ".join(rest[:rest.index("value")] if "value" in rest else rest)
            if name == "MultiPV":
                multipv = int(rest[rest.index("value") + 1])
        elif cmd == "position":
            if tokens[1] == "startpos":
                board = chess.Board()
                rest = tokens[2:]
            else:
                end = tokens.index("moves") if "moves" in tokens else len(tokens)
                board = chess.Board(" ".join(tokens[2:end]))
                rest = tokens[end:]
            if rest and rest[0] == "moves":
                for uci in rest[1:]:
                    board.push_uci(uci)
        elif cmd == "go":
            nodes = int(tokens[tokens.index("nodes") + 1]) if "nodes" in tokens else 1
            scored = sorted(((score_after(board, m), m) for m in board.legal_moves), key=sort_key, reverse=True)
            for i, ((kind, value), move) in enumerate(scored[:multipv], start=1):
                out.write(f"info depth 1 seldepth 1 multipv {i} score {kind} {value} nodes {nodes} "
                          f"nps 1000 time 1 pv {move.uci()}\n")
            out.write(f"bestmove {scored[0][1].uci() if scored else '0000'}\n")
        elif cmd == "quit":
            break
        out.flush()


if __name__ == "__main__":
    main()
