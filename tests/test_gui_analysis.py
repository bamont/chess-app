import pytest

pygame = pytest.importorskip("pygame")

from chess.pieces import Color  # noqa: E402
from review.model import Candidate, PlyAnalysis  # noqa: E402
from ui.gui import (  # noqa: E402
    ANALYSIS_BAR_HEIGHT,
    BAR_HEIGHT,
    BEST_ARROW_COLORS,
    BOARD_PIXELS,
    BUTTON_HEIGHT,
    BUTTON_WIDTH,
    BUTTON_X,
    BUTTON_Y,
    CURVE_HEIGHT,
    CURVE_LEFT,
    CURVE_TOP,
    CURVE_WIDTH,
    DETAIL_TOP,
    EVAL_BAR_WIDTH,
    LIST_ROWS,
    LIST_TOP,
    PANEL_WIDTH,
    PANEL_X,
    PLAYED_ARROW_COLORS,
    REVIEW_WINDOW_WIDTH,
    TEXT_WIDTH,
    WINDOW_HEIGHT,
    analysis_button_at,
    curve_hit,
    fit_text,
    navigate_review,
    request_analysis,
    review_arrows,
    review_click,
)
from ui.review_view import ROW_HEIGHT, ReviewSession  # noqa: E402


class FakeFont:
    def size(self, text):
        return len(text) * 10, 20


class FakeLauncher:
    def __init__(self):
        self.calls = []

    def start(self, game, status, white, black):
        self.calls.append((game, status, white, black))
        return True


class FakeController:
    def __init__(self, human_color):
        self.human_color = human_color
        self.game = "game"
        self.status = "status"


def test_the_window_has_a_bar_for_the_analysis():
    assert WINDOW_HEIGHT == BOARD_PIXELS + BAR_HEIGHT + ANALYSIS_BAR_HEIGHT


def test_the_button_sits_inside_the_analysis_bar():
    assert BUTTON_X >= 0
    assert BUTTON_X + BUTTON_WIDTH <= BOARD_PIXELS
    assert BUTTON_Y >= BOARD_PIXELS + BAR_HEIGHT
    assert BUTTON_Y + BUTTON_HEIGHT <= WINDOW_HEIGHT


def test_the_text_does_not_run_under_the_button():
    assert TEXT_WIDTH + 12 < BUTTON_X


def test_the_button_is_hit_only_inside_its_rectangle():
    assert analysis_button_at(BUTTON_X + BUTTON_WIDTH // 2, BUTTON_Y + BUTTON_HEIGHT // 2)
    assert analysis_button_at(BUTTON_X, BUTTON_Y)
    assert not analysis_button_at(BUTTON_X - 1, BUTTON_Y)
    assert not analysis_button_at(BUTTON_X + BUTTON_WIDTH, BUTTON_Y)
    assert not analysis_button_at(BUTTON_X, BUTTON_Y - 1)
    assert not analysis_button_at(BUTTON_X, BUTTON_Y + BUTTON_HEIGHT)


def test_the_board_and_the_status_bar_are_not_the_button():
    assert not analysis_button_at(10, 10)
    assert not analysis_button_at(BUTTON_X, BOARD_PIXELS + 5)


def test_a_short_text_is_left_alone():
    assert fit_text(FakeFont(), "ready", 100) == "ready"


def test_a_long_text_is_cut_with_an_ellipsis_that_fits():
    text = fit_text(FakeFont(), "a very long message that cannot fit", 100)
    assert text.endswith("...")
    assert len(text) * 10 <= 100
    assert text.startswith("a very")


def test_an_empty_text_stays_empty():
    assert fit_text(FakeFont(), "", 100) == ""


def test_the_analysis_is_requested_with_the_names_of_the_players():
    launcher = FakeLauncher()
    request_analysis(launcher, FakeController(Color.WHITE), "minimax:3")
    request_analysis(launcher, FakeController(Color.BLACK), "minimax:3")
    request_analysis(launcher, FakeController(None), "minimax:3")
    assert launcher.calls == [
        ("game", "status", "You", "minimax:3"),
        ("game", "status", "minimax:3", "You"),
        ("game", "status", "White", "Black"),
    ]


def make_ply(index, loss_cp=0, uci="e2e4", best="d2d4", candidates=()):
    return PlyAnalysis(index, f"fen{index}", "e4", uci, "d4", best, ("d4", "d5"), 30, 30 - loss_cp, tuple(candidates))


def review_of(count=5, **first):
    plies = [make_ply(0, **first)] + [make_ply(index) for index in range(1, count)]
    return ReviewSession(plies, make_board=lambda fen: fen)


def test_the_review_window_has_room_for_the_evaluation_bar_and_the_panel():
    assert PANEL_X == BOARD_PIXELS + EVAL_BAR_WIDTH
    assert REVIEW_WINDOW_WIDTH == BOARD_PIXELS + EVAL_BAR_WIDTH + PANEL_WIDTH
    assert LIST_TOP + LIST_ROWS * ROW_HEIGHT < DETAIL_TOP < WINDOW_HEIGHT


def test_arrow_keys_walk_through_the_review():
    session = review_of()
    assert navigate_review(session, pygame.K_RIGHT)
    assert session.cursor == 1
    assert navigate_review(session, pygame.K_DOWN)
    assert session.cursor == 2
    assert navigate_review(session, pygame.K_LEFT)
    assert navigate_review(session, pygame.K_UP)
    assert session.cursor == 0
    assert navigate_review(session, pygame.K_END)
    assert session.cursor == 4
    assert navigate_review(session, pygame.K_HOME)
    assert session.cursor == 0


def test_the_m_key_jumps_to_the_next_mistake():
    session = ReviewSession([make_ply(0), make_ply(1), make_ply(2, loss_cp=400)], make_board=lambda fen: fen)
    assert navigate_review(session, pygame.K_m)
    assert session.cursor == 2


def test_other_keys_do_nothing_in_the_review():
    session = review_of()
    assert not navigate_review(session, pygame.K_x)
    assert session.cursor == 0


def test_a_click_on_the_list_jumps_to_that_move():
    session = review_of()
    review_click(session, (PANEL_X + 50, LIST_TOP + 2 * ROW_HEIGHT + 3))
    assert session.cursor == 2


def test_a_click_outside_the_list_changes_nothing():
    session = review_of()
    review_click(session, (10, LIST_TOP + 2 * ROW_HEIGHT))
    review_click(session, (PANEL_X + 50, LIST_TOP - 1))
    review_click(session, (PANEL_X + 50, LIST_TOP + 20 * ROW_HEIGHT))
    review_click(session, (BOARD_PIXELS + EVAL_BAR_WIDTH // 2, LIST_TOP + 2 * ROW_HEIGHT))
    assert session.cursor == 0


def test_the_best_moves_are_green_with_decreasing_strength():
    candidates = [Candidate("d4", "d2d4", 30, ("d4",)), Candidate("Nf3", "g1f3", 25, ("Nf3",))]
    session = review_of(uci="h2h4", loss_cp=400, candidates=candidates)
    arrows = review_arrows(session)
    assert [color for _, color in arrows[:2]] == [BEST_ARROW_COLORS[0], BEST_ARROW_COLORS[1]]
    assert len(arrows) == 3


@pytest.mark.parametrize(
    "loss_cp, category",
    [(0, "excellent"), (75, "inaccuracy"), (140, "mistake"), (400, "blunder")],
)
def test_the_played_arrow_takes_the_color_of_the_move_quality(loss_cp, category):
    session = review_of(uci="h2h4", loss_cp=loss_cp)
    assert session.current.category == category
    assert review_arrows(session)[-1][1] == PLAYED_ARROW_COLORS[category]


def test_a_best_move_is_drawn_in_green():
    session = review_of(uci="d2d4", best="d2d4")
    assert review_arrows(session) == [(session.played_squares, BEST_ARROW_COLORS[0])]


def test_the_curve_fits_below_the_details_and_inside_the_panel():
    assert CURVE_TOP > DETAIL_TOP + 28 + 8 * 20
    assert CURVE_TOP + CURVE_HEIGHT <= WINDOW_HEIGHT
    assert CURVE_LEFT >= PANEL_X
    assert CURVE_LEFT + CURVE_WIDTH <= PANEL_X + PANEL_WIDTH


def test_the_curve_is_hit_only_inside_its_rectangle():
    assert curve_hit((CURVE_LEFT, CURVE_TOP))
    assert curve_hit((CURVE_LEFT + CURVE_WIDTH, CURVE_TOP + CURVE_HEIGHT))
    assert not curve_hit((CURVE_LEFT - 1, CURVE_TOP))
    assert not curve_hit((CURVE_LEFT, CURVE_TOP - 1))
    assert not curve_hit((CURVE_LEFT, CURVE_TOP + CURVE_HEIGHT + 1))


def test_a_click_on_the_curve_jumps_to_that_move():
    session = review_of(count=5)
    review_click(session, (CURVE_LEFT + CURVE_WIDTH, CURVE_TOP + 10))
    assert session.cursor == 4
    review_click(session, (CURVE_LEFT, CURVE_TOP + 10))
    assert session.cursor == 0
    review_click(session, (CURVE_LEFT + CURVE_WIDTH // 2, CURVE_TOP + 10))
    assert session.cursor == 2