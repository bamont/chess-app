import queue
import threading

from bot.base import Bot
from chess.board import Board
from chess.move import Move


class BotPlayer:
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self._results: queue.Queue = queue.Queue()
        self._thinking = False

    @property
    def thinking(self) -> bool:
        return self._thinking

    def start(self, board: Board) -> None:
        if self._thinking:
            return
        self._thinking = True
        snapshot = board.copy()
        threading.Thread(target=self._work, args=(snapshot,), daemon=True).start()

    def poll(self) -> Move | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._thinking = False
        if isinstance(result, Exception):
            raise result
        return result

    def _work(self, board: Board) -> None:
        try:
            self._results.put(self.bot.choose_move(board))
        except Exception as error:
            self._results.put(error)