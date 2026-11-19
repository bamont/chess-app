from collections.abc import Callable

from bot.advanced_evaluation import evaluate_advanced, evaluate_advanced_without_shield
from bot.base import Bot
from bot.evaluation import evaluate, evaluate_positional
from bot.greedy_bot import GreedyBot
from bot.minimax_bot import MinimaxBot
from bot.opening_book import default_book
from bot.random_bot import RandomBot

EVALUATORS = {
    "material": evaluate,
    "positional": evaluate_positional,
    "advanced": evaluate_advanced,
    "noshield": evaluate_advanced_without_shield,
}
DEFAULT_DEPTH = 2
LIMITED_SEARCH_DEPTH = 64
DEFAULT_EVALUATOR = "advanced"
QUIESCENCE_LIMIT = 6


def create_bot(spec: str, seed: int | None = None) -> Bot:
    name, *options = spec.split(":")
    if name == "random" and not options:
        return RandomBot(seed)
    if name == "greedy" and not options:
        return GreedyBot(seed)
    if name == "minimax":
        return _create_minimax(spec, options, seed)
    raise ValueError(f"Unknown bot specification: {spec}")


def _parse_value(option: str, spec: str, cast: Callable):
    try:
        return cast(option.split("=", 1)[1])
    except ValueError:
        raise ValueError(f"Invalid value in option '{option}' of bot specification: {spec}") from None


def _create_minimax(spec: str, options: list[str], seed: int | None) -> MinimaxBot:
    depth = None
    evaluator_name = DEFAULT_EVALUATOR
    use_book = True
    quiescence_limit = QUIESCENCE_LIMIT
    repetition_aware = True
    use_tt = True
    iterative = True
    null_move = True
    lmr = True
    check_extension = True
    time_limit = None
    node_limit = None
    for option in options:
        if option.isdigit():
            depth = int(option)
        elif option in EVALUATORS:
            evaluator_name = option
        elif option == "nobook":
            use_book = False
        elif option == "noquiescence":
            quiescence_limit = 0
        elif option.startswith("quiescence="):
            quiescence_limit = _parse_value(option, spec, int)
            if quiescence_limit < 0:
                raise ValueError(f"Invalid value in option '{option}' of bot specification: {spec}")
        elif option == "norepetition":
            repetition_aware = False
        elif option == "notranspositions":
            use_tt = False
        elif option == "noiterative":
            iterative = False
        elif option == "nonullmove":
            null_move = False
        elif option == "nolmr":
            lmr = False
        elif option == "nocheckext":
            check_extension = False
        elif option.startswith("time="):
            time_limit = _parse_value(option, spec, float)
        elif option.startswith("nodes="):
            node_limit = _parse_value(option, spec, int)
        else:
            raise ValueError(f"Unknown option '{option}' in bot specification: {spec}")
    if depth is None:
        limited = time_limit is not None or node_limit is not None
        depth = LIMITED_SEARCH_DEPTH if limited else DEFAULT_DEPTH
    return MinimaxBot(
        depth=depth,
        seed=seed,
        evaluator=EVALUATORS[evaluator_name],
        book=default_book() if use_book else None,
        quiescence_limit=quiescence_limit,
        repetition_aware=repetition_aware,
        use_tt=use_tt,
        iterative=iterative,
        time_limit=time_limit,
        node_limit=node_limit,
        null_move=null_move,
        lmr=lmr,
        check_extension=check_extension,
    )