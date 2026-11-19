import pytest

from bot.advanced_evaluation import evaluate_advanced, evaluate_advanced_without_shield
from bot.evaluation import evaluate
from bot.factory import create_bot
from bot.greedy_bot import GreedyBot
from bot.minimax_bot import MinimaxBot
from bot.random_bot import RandomBot


def test_simple_bots():
    assert isinstance(create_bot("random"), RandomBot)
    assert isinstance(create_bot("greedy"), GreedyBot)


def test_minimax_defaults():
    bot = create_bot("minimax")
    assert isinstance(bot, MinimaxBot)
    assert bot.depth == 2
    assert bot.evaluator is evaluate_advanced
    assert bot.book is not None
    assert bot.quiescence_limit == 6


def test_minimax_options():
    bot = create_bot("minimax:3:material")
    assert bot.depth == 3
    assert bot.evaluator is evaluate


def test_minimax_options_in_any_order():
    bot = create_bot("minimax:material:3")
    assert bot.depth == 3
    assert bot.evaluator is evaluate


def test_minimax_can_disable_book_and_quiescence():
    bot = create_bot("minimax:nobook:noquiescence")
    assert bot.book is None
    assert bot.quiescence_limit == 0


@pytest.mark.parametrize(
    "spec",
    ["", "chess", "random:3", "minimax:abc", "minimax:2:fancy", "minimax:2:material:extra", "minimax:0"],
)
def test_invalid_specifications_raise(spec):
    with pytest.raises(ValueError):
        create_bot(spec)


def test_minimax_can_disable_repetition_detection():
    assert create_bot("minimax").repetition_aware is True
    assert create_bot("minimax:norepetition").repetition_aware is False


def test_time_and_node_limits():
    timed = create_bot("minimax:time=0.5")
    assert timed.time_limit == 0.5
    assert timed.depth == 64
    limited = create_bot("minimax:3:nodes=1000")
    assert limited.depth == 3
    assert limited.node_limit == 1000


def test_search_features_can_be_disabled():
    bot = create_bot("minimax:notranspositions:noiterative")
    assert bot.use_tt is False
    assert bot.iterative is False


@pytest.mark.parametrize("spec", ["minimax:time=abc", "minimax:time=0", "minimax:nodes=1.5", "minimax:nodes=0"])
def test_invalid_limits_raise(spec):
    with pytest.raises(ValueError):
        create_bot(spec)


def test_pruning_techniques_are_enabled_by_default():
    bot = create_bot("minimax")
    assert bot.null_move is True
    assert bot.lmr is True


def test_pruning_techniques_can_be_disabled():
    bot = create_bot("minimax:nonullmove:nolmr")
    assert bot.null_move is False
    assert bot.lmr is False


@pytest.mark.parametrize("spec", ["minimax:time=1:noiterative", "minimax:nodes=100:noiterative"])
def test_limits_cannot_be_combined_with_noiterative(spec):
    with pytest.raises(ValueError):
        create_bot(spec)


def test_advanced_evaluator_can_be_selected():
    assert create_bot("minimax:advanced").evaluator is evaluate_advanced


def test_check_extension_is_enabled_by_default():
    assert create_bot("minimax").check_extension is True


def test_check_extension_can_be_disabled():
    assert create_bot("minimax:nocheckext").check_extension is False


def test_the_default_evaluator_includes_the_shield():
    assert create_bot("minimax").evaluator is evaluate_advanced


def test_the_shield_can_be_disabled():
    assert create_bot("minimax:noshield").evaluator is evaluate_advanced_without_shield


def test_the_quiescence_limit_can_be_set():
    create_bot("minimax:quiescence=3")
    create_bot("minimax:quiescence=0")


def test_an_invalid_quiescence_limit_is_rejected():
    with pytest.raises(ValueError):
        create_bot("minimax:quiescence=x")
    with pytest.raises(ValueError):
        create_bot("minimax:quiescence=-1")