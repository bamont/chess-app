from pathlib import Path

import pygame

from bot.base import Bot
from chess.board import Board
from chess.game import Game
from chess.move import Square
from chess.pieces import Color, Kind
from ui.bot_player import BotPlayer
from ui.controller import BoardController
from ui.eval_bar import bar_label, label_at_top, white_region
from ui.eval_curve import curve_index_at, curve_points, curve_values, curve_x
from ui.messages import describe_hint, describe_status
from ui.options import resolve_color
from ui.review_job import ReviewLauncher, player_names
from ui.review_view import ROW_HEIGHT, ReviewSession, arrow_shape, row_at, visible_start

SQUARE_SIZE = 80
BOARD_PIXELS = 8 * SQUARE_SIZE
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
ASSETS_DIR = Path(__file__).parent / "assets" / "pieces"
FPS = 60

SELECTED_COLOR = (246, 246, 105, 150)
LAST_MOVE_COLOR = (205, 210, 106, 120)
HINT_COLOR = (20, 85, 30, 110)

BAR_HEIGHT = 40
BAR_COLOR = (38, 36, 33)
TEXT_COLOR = (230, 230, 230)
PROMOTION_CHOICES = (Kind.QUEEN, Kind.ROOK, Kind.BISHOP, Kind.KNIGHT)
PICKER_X = (BOARD_PIXELS - len(PROMOTION_CHOICES) * SQUARE_SIZE) // 2
PICKER_Y = (BOARD_PIXELS - SQUARE_SIZE) // 2

ANALYSIS_BAR_HEIGHT = 36
WINDOW_HEIGHT = BOARD_PIXELS + BAR_HEIGHT + ANALYSIS_BAR_HEIGHT
ANALYSIS_BAR_COLOR = (30, 29, 27)
BUTTON_WIDTH = 130
BUTTON_HEIGHT = 26
BUTTON_X = BOARD_PIXELS - BUTTON_WIDTH - 12
BUTTON_Y = BOARD_PIXELS + BAR_HEIGHT + (ANALYSIS_BAR_HEIGHT - BUTTON_HEIGHT) // 2
BUTTON_COLOR = (74, 117, 76)
BUTTON_DISABLED_COLOR = (88, 88, 88)
TEXT_WIDTH = BUTTON_X - 24
REVIEWS_DIR = Path("reviews")

PANEL_WIDTH = 320
EVAL_BAR_WIDTH = 44
PANEL_X = BOARD_PIXELS + EVAL_BAR_WIDTH
REVIEW_WINDOW_WIDTH = PANEL_X + PANEL_WIDTH
EVAL_WHITE_COLOR = (238, 238, 238)
EVAL_BLACK_COLOR = (58, 56, 54)
EVAL_MIDDLE_COLOR = (130, 130, 130)
CURVE_HEIGHT = 70
CURVE_TOP = WINDOW_HEIGHT - CURVE_HEIGHT - 14
CURVE_LEFT = PANEL_X + 12
CURVE_WIDTH = PANEL_WIDTH - 24
CURVE_CURSOR_COLOR = (240, 200, 60)
PANEL_COLOR = (38, 36, 33)
PANEL_CURRENT_COLOR = (72, 68, 62)
PANEL_MUTED_COLOR = (150, 150, 150)
LIST_TOP = 92
LIST_ROWS = 14
DETAIL_TOP = LIST_TOP + LIST_ROWS * ROW_HEIGHT + 10
PLAYED_ARROW_COLORS = {
    "best": (46, 139, 87, 210),
    "excellent": (66, 133, 196, 200),
    "good": (66, 133, 196, 200),
    "inaccuracy": (201, 162, 39, 200),
    "mistake": (224, 128, 30, 200),
    "blunder": (214, 69, 69, 200),
}
BEST_ARROW_COLORS = ((46, 139, 87, 210), (46, 139, 87, 140), (46, 139, 87, 90))
CATEGORY_COLORS = {
    "best": (46, 139, 87),
    "excellent": (74, 159, 120),
    "good": (138, 168, 107),
    "inaccuracy": (201, 162, 39),
    "mistake": (224, 128, 30),
    "blunder": (214, 69, 69),
}

PieceImages = dict[tuple[Color, Kind], pygame.Surface]


def square_to_pixel(square: Square, flipped: bool = False) -> tuple[int, int]:
    row, col = square
    if flipped:
        row, col = 7 - row, 7 - col
    return col * SQUARE_SIZE, row * SQUARE_SIZE


def pixel_to_square(x: int, y: int, flipped: bool = False) -> Square | None:
    if not (0 <= x < BOARD_PIXELS and 0 <= y < BOARD_PIXELS):
        return None
    row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
    return (7 - row, 7 - col) if flipped else (row, col)


def draw_board(screen: pygame.Surface, flipped: bool) -> None:
    for row in range(8):
        for col in range(8):
            color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
            x, y = square_to_pixel((row, col), flipped)
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))


def find_piece_file(name: str) -> Path:
    for extension in (".svg", ".png"):
        path = ASSETS_DIR / f"{name}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing piece image '{name}' in {ASSETS_DIR}")


def load_piece_image(path: Path, size: int) -> pygame.Surface:
    if path.suffix == ".svg":
        return pygame.image.load_sized_svg(str(path), (size, size)).convert_alpha()
    image = pygame.image.load(str(path)).convert_alpha()
    return pygame.transform.smoothscale(image, (size, size))


def load_piece_images(square_size: int) -> PieceImages:
    images = {}
    for color in Color:
        for kind in Kind:
            path = find_piece_file(f"{color.value}{kind.value}")
            images[(color, kind)] = load_piece_image(path, square_size)
    return images


def fill_square(screen: pygame.Surface, square: Square, color: tuple[int, int, int, int], flipped: bool) -> None:
    overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    overlay.fill(color)
    screen.blit(overlay, square_to_pixel(square, flipped))


def draw_highlights(screen: pygame.Surface, controller: BoardController, flipped: bool) -> None:
    history = controller.game.board.history
    if history:
        last_move = history[-1].move
        fill_square(screen, last_move.start, LAST_MOVE_COLOR, flipped)
        fill_square(screen, last_move.end, LAST_MOVE_COLOR, flipped)
    if controller.selected is not None:
        fill_square(screen, controller.selected, SELECTED_COLOR, flipped)


def draw_pieces(screen: pygame.Surface, board: Board, images: PieceImages, flipped: bool) -> None:
    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is not None:
                screen.blit(images[(piece.color, piece.kind)], square_to_pixel((row, col), flipped))


def draw_destinations(screen: pygame.Surface, controller: BoardController, flipped: bool) -> None:
    center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
    for square in controller.destinations:
        overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        if controller.game.board.piece_at(square) is None:
            pygame.draw.circle(overlay, HINT_COLOR, center, SQUARE_SIZE // 6)
        else:
            pygame.draw.circle(overlay, HINT_COLOR, center, SQUARE_SIZE // 2 - 4, 6)
        screen.blit(overlay, square_to_pixel(square, flipped))


def promotion_choice_at(x: int, y: int) -> Kind | None:
    if not (PICKER_Y <= y < PICKER_Y + SQUARE_SIZE):
        return None
    index = (x - PICKER_X) // SQUARE_SIZE
    if 0 <= index < len(PROMOTION_CHOICES):
        return PROMOTION_CHOICES[index]
    return None


def draw_promotion_picker(screen: pygame.Surface, color: Color, images: PieceImages) -> None:
    dim = pygame.Surface((BOARD_PIXELS, BOARD_PIXELS), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 140))
    screen.blit(dim, (0, 0))
    width = len(PROMOTION_CHOICES) * SQUARE_SIZE
    pygame.draw.rect(screen, LIGHT_COLOR, (PICKER_X, PICKER_Y, width, SQUARE_SIZE))
    for index, kind in enumerate(PROMOTION_CHOICES):
        screen.blit(images[(color, kind)], (PICKER_X + index * SQUARE_SIZE, PICKER_Y))


def draw_status_bar(screen: pygame.Surface, font: pygame.font.Font, text: str, hint: str) -> None:
    pygame.draw.rect(screen, BAR_COLOR, (0, BOARD_PIXELS, BOARD_PIXELS, BAR_HEIGHT))
    label = font.render(text, True, TEXT_COLOR)
    y = BOARD_PIXELS + (BAR_HEIGHT - label.get_height()) // 2
    screen.blit(label, (12, y))
    hint_label = font.render(hint, True, TEXT_COLOR)
    screen.blit(hint_label, (BOARD_PIXELS - hint_label.get_width() - 12, y))


def analysis_button_at(x: int, y: int) -> bool:
    return BUTTON_X <= x < BUTTON_X + BUTTON_WIDTH and BUTTON_Y <= y < BUTTON_Y + BUTTON_HEIGHT


def fit_text(font: pygame.font.Font, text: str, width: int) -> str:
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "...")[0] > width:
        text = text[:-1]
    return text + "..."


def draw_analysis_bar(
    screen: pygame.Surface, font: pygame.font.Font, text: str, enabled: bool, caption: str = "Analyse (A)"
) -> None:
    top = BOARD_PIXELS + BAR_HEIGHT
    pygame.draw.rect(screen, ANALYSIS_BAR_COLOR, (0, top, BOARD_PIXELS, ANALYSIS_BAR_HEIGHT))
    label = font.render(fit_text(font, text, TEXT_WIDTH), True, TEXT_COLOR)
    screen.blit(label, (12, top + (ANALYSIS_BAR_HEIGHT - label.get_height()) // 2))
    color = BUTTON_COLOR if enabled else BUTTON_DISABLED_COLOR
    pygame.draw.rect(screen, color, (BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=6)
    label_caption = font.render(caption, True, TEXT_COLOR)
    caption_x = BUTTON_X + (BUTTON_WIDTH - label_caption.get_width()) // 2
    screen.blit(label_caption, (caption_x, BUTTON_Y + (BUTTON_HEIGHT - label_caption.get_height()) // 2))


def square_center(square: Square, flipped: bool) -> tuple[float, float]:
    x, y = square_to_pixel(square, flipped)
    return x + SQUARE_SIZE / 2, y + SQUARE_SIZE / 2


def review_arrows(session: ReviewSession) -> list[tuple[tuple[Square, Square], tuple[int, int, int, int]]]:
    last = len(BEST_ARROW_COLORS) - 1
    shapes = [(squares, BEST_ARROW_COLORS[min(index, last)]) for index, squares in enumerate(session.best_squares)]
    played = session.played_squares
    if played is not None:
        shapes.append((played, PLAYED_ARROW_COLORS.get(session.current.category, PLAYED_ARROW_COLORS["good"])))
    return shapes


def draw_arrows(screen: pygame.Surface, shapes: list, flipped: bool) -> None:
    overlay = pygame.Surface((BOARD_PIXELS, BOARD_PIXELS), pygame.SRCALPHA)
    for (start, end), color in shapes:
        shape = arrow_shape(square_center(start, flipped), square_center(end, flipped), SQUARE_SIZE)
        if shape is None:
            continue
        shaft_start, shaft_end, head = shape
        pygame.draw.line(overlay, color, shaft_start, shaft_end, SQUARE_SIZE // 7)
        pygame.draw.polygon(overlay, color, head)
    screen.blit(overlay, (0, 0))


def draw_eval_bar(screen: pygame.Surface, font: pygame.font.Font, white_cp: int, flipped: bool) -> None:
    pygame.draw.rect(screen, PANEL_COLOR, (BOARD_PIXELS, BOARD_PIXELS, EVAL_BAR_WIDTH, WINDOW_HEIGHT - BOARD_PIXELS))
    pygame.draw.rect(screen, EVAL_BLACK_COLOR, (BOARD_PIXELS, 0, EVAL_BAR_WIDTH, BOARD_PIXELS))
    top, size = white_region(white_cp, BOARD_PIXELS, flipped)
    pygame.draw.rect(screen, EVAL_WHITE_COLOR, (BOARD_PIXELS, top, EVAL_BAR_WIDTH, size))
    middle = BOARD_PIXELS // 2
    pygame.draw.line(screen, EVAL_MIDDLE_COLOR, (BOARD_PIXELS, middle), (BOARD_PIXELS + EVAL_BAR_WIDTH, middle))
    label = font.render(bar_label(white_cp), True, EVAL_BLACK_COLOR if white_cp >= 0 else EVAL_WHITE_COLOR)
    y = 6 if label_at_top(white_cp, flipped) else BOARD_PIXELS - label.get_height() - 6
    screen.blit(label, (BOARD_PIXELS + (EVAL_BAR_WIDTH - label.get_width()) // 2, y))


def curve_hit(position: tuple[int, int]) -> bool:
    return CURVE_LEFT <= position[0] <= CURVE_LEFT + CURVE_WIDTH and CURVE_TOP <= position[1] <= CURVE_TOP + CURVE_HEIGHT


def draw_eval_curve(screen: pygame.Surface, session: ReviewSession, flipped: bool) -> None:
    values = curve_values(session.evaluations())
    points = curve_points(values, CURVE_LEFT, CURVE_TOP, CURVE_WIDTH, CURVE_HEIGHT, flipped)
    pygame.draw.rect(screen, EVAL_BLACK_COLOR, (CURVE_LEFT, CURVE_TOP, CURVE_WIDTH, CURVE_HEIGHT))
    base = CURVE_TOP if flipped else CURVE_TOP + CURVE_HEIGHT
    if len(points) > 1:
        pygame.draw.polygon(screen, EVAL_WHITE_COLOR, [(CURVE_LEFT, base)] + points + [(CURVE_LEFT + CURVE_WIDTH, base)])
    middle = CURVE_TOP + CURVE_HEIGHT // 2
    pygame.draw.line(screen, EVAL_MIDDLE_COLOR, (CURVE_LEFT, middle), (CURVE_LEFT + CURVE_WIDTH, middle))
    for move, point in zip(session.moves, points, strict=True):
        if move.category in ("mistake", "blunder"):
            pygame.draw.circle(screen, CATEGORY_COLORS[move.category], (round(point[0]), round(point[1])), 3)
    cursor = round(curve_x(session.cursor, CURVE_LEFT, CURVE_WIDTH, len(session)))
    pygame.draw.line(screen, CURVE_CURSOR_COLOR, (cursor, CURVE_TOP), (cursor, CURVE_TOP + CURVE_HEIGHT), 2)


def draw_review_panel(screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font, session: ReviewSession) -> None:
    pygame.draw.rect(screen, PANEL_COLOR, (PANEL_X, 0, PANEL_WIDTH, WINDOW_HEIGHT))
    x = PANEL_X + 12
    width = PANEL_WIDTH - 24
    screen.blit(font.render("Review", True, TEXT_COLOR), (x, 10))
    for index, text in enumerate(("Left/Right: move   M: next mistake", "Home/End: first/last   H: report   Esc: exit")):
        screen.blit(small_font.render(fit_text(small_font, text, width), True, PANEL_MUTED_COLOR), (x, 38 + index * 18))
    start = visible_start(session.cursor, len(session), LIST_ROWS)
    for offset in range(min(LIST_ROWS, len(session) - start)):
        index = start + offset
        row = session.row(index)
        top = LIST_TOP + offset * ROW_HEIGHT
        if index == session.cursor:
            pygame.draw.rect(screen, PANEL_CURRENT_COLOR, (PANEL_X + 4, top, PANEL_WIDTH - 8, ROW_HEIGHT))
        pygame.draw.circle(screen, CATEGORY_COLORS[row.category], (x + 4, top + ROW_HEIGHT // 2), 4)
        screen.blit(small_font.render(row.text, True, TEXT_COLOR), (x + 16, top + 3))
        if row.best:
            label = small_font.render(fit_text(small_font, row.best, 110), True, PANEL_MUTED_COLOR)
            screen.blit(label, (PANEL_X + PANEL_WIDTH - label.get_width() - 12, top + 3))
    y = DETAIL_TOP
    headline = font.render(fit_text(font, session.headline(), width), True, CATEGORY_COLORS[session.current.category])
    screen.blit(headline, (x, y))
    y += 28
    for text in session.detail_lines():
        screen.blit(small_font.render(fit_text(small_font, text, width), True, TEXT_COLOR), (x, y))
        y += 20


def draw_review(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    images: PieceImages,
    session: ReviewSession,
    flipped: bool,
) -> None:
    draw_board(screen, flipped)
    played = session.played_squares
    if played is not None:
        for square in played:
            fill_square(screen, square, LAST_MOVE_COLOR, flipped)
    draw_pieces(screen, session.board, images, flipped)
    draw_arrows(screen, review_arrows(session), flipped)
    draw_eval_bar(screen, small_font, session.current.white_cp_before, flipped)
    draw_status_bar(screen, font, session.headline(), "Esc: exit review")
    draw_analysis_bar(screen, font, f"Move {session.cursor + 1} of {len(session)}", True, "Exit review")
    draw_review_panel(screen, font, small_font, session)
    draw_eval_curve(screen, session, flipped)


REVIEW_NAVIGATION = {
    pygame.K_RIGHT: ReviewSession.next,
    pygame.K_DOWN: ReviewSession.next,
    pygame.K_LEFT: ReviewSession.previous,
    pygame.K_UP: ReviewSession.previous,
    pygame.K_HOME: ReviewSession.first,
    pygame.K_END: ReviewSession.last,
    pygame.K_m: ReviewSession.next_mistake,
}


def navigate_review(session: ReviewSession, key: int) -> bool:
    action = REVIEW_NAVIGATION.get(key)
    if action is None:
        return False
    action(session)
    return True


def review_click(session: ReviewSession, position: tuple[int, int]) -> None:
    if curve_hit(position):
        index = curve_index_at(position[0], CURVE_LEFT, CURVE_WIDTH, len(session))
        if index is not None:
            session.go_to(index)
        return
    if position[0] < PANEL_X:
        return
    start = visible_start(session.cursor, len(session), LIST_ROWS)
    index = row_at(position[1], LIST_TOP, start, LIST_ROWS, len(session))
    if index is not None:
        session.go_to(index)


def open_review(plies: list) -> tuple[ReviewSession, pygame.Surface]:
    session = ReviewSession(plies)
    return session, pygame.display.set_mode((REVIEW_WINDOW_WIDTH, WINDOW_HEIGHT))


def close_review() -> pygame.Surface:
    return pygame.display.set_mode((BOARD_PIXELS, WINDOW_HEIGHT))


def request_analysis(launcher: ReviewLauncher, controller: BoardController, bot_name: str) -> None:
    white, black = player_names(controller.human_color, bot_name)
    launcher.start(controller.game, controller.status, white, black)


def handle_click(controller: BoardController, position: tuple[int, int], flipped: bool) -> None:
    if controller.pending_promotion:
        kind = promotion_choice_at(*position)
        if kind is None:
            controller.cancel_promotion()
        else:
            controller.choose_promotion(kind)
        return
    square = pixel_to_square(*position, flipped)
    if square is not None:
        controller.click(square)


def new_game(bot: Bot | None, color_choice: str) -> tuple[BoardController, BotPlayer | None]:
    if bot is None:
        return BoardController(Game()), None
    return BoardController(Game(), resolve_color(color_choice)), BotPlayer(bot)


def main(
    bot: Bot | None = None,
    color_choice: str = "white",
    bot_name: str = "Bot",
    engine: str | None = None,
    motifs: str | None = None,
    analysis_depth: int = 12,
    analysis_lines: int = 3,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((BOARD_PIXELS, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess")
    images = load_piece_images(SQUARE_SIZE)
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    controller, bot_player = new_game(bot, color_choice)
    launcher = ReviewLauncher(engine, analysis_depth, motifs, REVIEWS_DIR, lines=analysis_lines)
    review: ReviewSession | None = None
    clock = pygame.time.Clock()
    running = True
    while running:
        flipped = controller.human_color == Color.BLACK
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if review is not None:
                    if event.key == pygame.K_ESCAPE:
                        review = None
                        screen = close_review()
                    elif event.key == pygame.K_h:
                        launcher.open_report()
                    else:
                        navigate_review(review, event.key)
                elif event.key == pygame.K_n:
                    controller, bot_player = new_game(bot, color_choice)
                    launcher.reset()
                    flipped = controller.human_color == Color.BLACK
                elif event.key == pygame.K_a:
                    request_analysis(launcher, controller, bot_name)
                elif event.key == pygame.K_r and launcher.plies is not None:
                    review, screen = open_review(launcher.plies)
                elif event.key == pygame.K_h:
                    launcher.open_report()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if analysis_button_at(*event.pos):
                    if review is not None:
                        review = None
                        screen = close_review()
                    else:
                        request_analysis(launcher, controller, bot_name)
                elif review is not None:
                    review_click(review, event.pos)
                else:
                    handle_click(controller, event.pos, flipped)
        if review is None:
            fresh = launcher.take_fresh()
            if fresh is not None:
                review, screen = open_review(fresh)
        if bot_player is not None and review is None:
            if controller.is_bot_turn and not bot_player.thinking:
                bot_player.start(controller.game.board)
            move = bot_player.poll()
            if move is not None:
                controller.play_move(move)
        board = controller.game.board
        if review is not None:
            draw_review(screen, font, small_font, images, review, flipped)
        else:
            draw_board(screen, flipped)
            draw_highlights(screen, controller, flipped)
            draw_pieces(screen, board, images, flipped)
            draw_destinations(screen, controller, flipped)
            if controller.pending_promotion:
                draw_promotion_picker(screen, board.turn, images)
            if bot_player is not None and bot_player.thinking:
                text = "Bot is thinking..."
            else:
                text = describe_status(controller.status, board.turn, controller.in_check)
            draw_status_bar(screen, font, text, describe_hint(controller.human_color))
            draw_analysis_bar(screen, font, launcher.status(), launcher.available and not launcher.busy)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()