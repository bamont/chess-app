from chess.board import Board
from chess.game import Game, GameStatus
from chess.pieces import Color, Kind, Piece
from chess.utils import to_square
from ui.controller import BoardController


def controller_from_fen(fen: str, human_color: Color | None = None) -> BoardController:
    return BoardController(Game(Board.from_fen(fen)), human_color)


def click(controller: BoardController, *squares: str) -> None:
    for square in squares:
        controller.click(to_square(square))


def test_selecting_own_piece_shows_destinations():
    controller = BoardController(Game())
    click(controller, "e2")
    assert controller.selected == to_square("e2")
    assert controller.destinations == {to_square("e3"), to_square("e4")}


def test_empty_square_without_selection_does_nothing():
    controller = BoardController(Game())
    click(controller, "e4")
    assert controller.selected is None


def test_cannot_select_opponent_piece():
    controller = BoardController(Game())
    click(controller, "e7")
    assert controller.selected is None


def test_click_on_destination_plays_the_move():
    controller = BoardController(Game())
    click(controller, "e2", "e4")
    board = controller.game.board
    assert board.piece_at(to_square("e4")) == Piece(Color.WHITE, Kind.PAWN)
    assert board.turn == Color.BLACK
    assert controller.selected is None
    assert len(board.history) == 1


def test_clicking_selected_square_again_deselects():
    controller = BoardController(Game())
    click(controller, "e2", "e2")
    assert controller.selected is None


def test_clicking_another_own_piece_changes_selection():
    controller = BoardController(Game())
    click(controller, "e2", "g1")
    assert controller.selected == to_square("g1")
    assert controller.destinations == {to_square("f3"), to_square("h3")}


def test_clicking_illegal_square_deselects_without_moving():
    controller = BoardController(Game())
    click(controller, "e2", "e5")
    assert controller.selected is None
    assert controller.game.board.history == []


def test_capture():
    controller = controller_from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    click(controller, "e4", "d5")
    board = controller.game.board
    assert board.piece_at(to_square("d5")) == Piece(Color.WHITE, Kind.PAWN)
    assert board.piece_at(to_square("e4")) is None


def test_castling_by_clicking_the_king():
    controller = controller_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    click(controller, "e1", "g1")
    board = controller.game.board
    assert board.piece_at(to_square("g1")) == Piece(Color.WHITE, Kind.KING)
    assert board.piece_at(to_square("f1")) == Piece(Color.WHITE, Kind.ROOK)


def test_en_passant_by_clicking():
    controller = controller_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    click(controller, "e5", "d6")
    assert controller.game.board.piece_at(to_square("d5")) is None


def test_promotion_waits_for_the_choice():
    controller = controller_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    click(controller, "a7", "a8")
    assert len(controller.pending_promotion) == 4
    assert controller.game.board.piece_at(to_square("a7")) == Piece(Color.WHITE, Kind.PAWN)
    controller.choose_promotion(Kind.KNIGHT)
    assert controller.game.board.piece_at(to_square("a8")) == Piece(Color.WHITE, Kind.KNIGHT)
    assert controller.pending_promotion == []


def test_clicks_are_ignored_while_promotion_is_pending():
    controller = controller_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    click(controller, "a7", "a8", "e1")
    assert len(controller.pending_promotion) == 4
    assert controller.game.board.history == []


def test_status_is_updated_after_each_move():
    controller = BoardController(Game())
    assert controller.status == GameStatus.ONGOING
    click(controller, "f2", "f3", "e7", "e5", "g2", "g4", "d8", "h4")
    assert controller.status == GameStatus.BLACK_WINS


def test_clicks_are_ignored_when_the_game_is_over():
    controller = BoardController(Game())
    click(controller, "f2", "f3", "e7", "e5", "g2", "g4", "d8", "h4")
    click(controller, "e1")
    assert controller.selected is None


def test_clicks_are_ignored_on_the_bots_turn():
    controller = BoardController(Game(), human_color=Color.BLACK)
    click(controller, "e2")
    assert controller.selected is None


def test_cancel_promotion_keeps_the_pawn_on_its_square():
    controller = controller_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    click(controller, "a7", "a8")
    controller.cancel_promotion()
    assert controller.pending_promotion == []
    assert controller.selected is None
    assert controller.game.board.history == []


def test_in_check_flag_follows_the_position():
    controller = controller_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert not controller.in_check
    click(controller, "a1", "a8")
    assert controller.in_check
    assert controller.status == GameStatus.ONGOING


def test_is_bot_turn():
    assert BoardController(Game(), human_color=Color.BLACK).is_bot_turn
    assert not BoardController(Game(), human_color=Color.WHITE).is_bot_turn
    assert not BoardController(Game()).is_bot_turn


def test_play_move_applies_an_external_move():
    game = Game()
    controller = BoardController(game, human_color=Color.BLACK)
    move = next(m for m in game.legal_moves() if m.start == to_square("e2") and m.end == to_square("e4"))
    controller.play_move(move)
    assert game.board.piece_at(to_square("e4")) == Piece(Color.WHITE, Kind.PAWN)
    assert not controller.is_bot_turn


def test_no_bot_turn_once_the_game_is_over():
    controller = BoardController(Game(), human_color=Color.BLACK)
    for notation in ("f2f3", "e7e5", "g2g4", "d8h4"):
        start, end = to_square(notation[:2]), to_square(notation[2:])
        move = next(m for m in controller.game.legal_moves() if m.start == start and m.end == end)
        controller.play_move(move)
    assert controller.status == GameStatus.BLACK_WINS
    assert not controller.is_bot_turn