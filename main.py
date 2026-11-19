import argparse
from pathlib import Path

from bot.factory import create_bot
from ui.options import COLOR_CHOICES, resolve_color
from ui.review_job import resolve_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play chess.")
    parser.add_argument("--bot", help="opponent: random, greedy or minimax[:depth[:material|positional]]")
    parser.add_argument("--color", choices=COLOR_CHOICES, default="white", help="your color against the bot")
    parser.add_argument("--gui", action="store_true", help="use the graphical interface")
    parser.add_argument("--engine", help="UCI engine, such as Stockfish, used by the Analyse button of the graphical interface")
    parser.add_argument("--motifs", help="NumPy motif model that adds tactical themes to the analysis")
    parser.add_argument("--analysis-depth", type=int, default=12, help="engine depth used by the Analyse button")
    parser.add_argument("--analysis-lines", type=int, default=3, help="number of best moves shown for every move of the review")
    return parser


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()
    bot = None
    if args.bot:
        try:
            bot = create_bot(args.bot)
        except ValueError as error:
            parser.error(str(error))
    if (args.engine or args.motifs) and not args.gui:
        parser.error("The analysis is only available in the graphical interface: add --gui")
    engine = None
    if args.engine:
        engine = resolve_engine(args.engine)
        if engine is None:
            parser.error(f"Engine not found: {args.engine}")
    if args.motifs and not Path(args.motifs).is_file():
        parser.error(f"Motif model not found: {args.motifs}")
    if args.gui:
        from ui.gui import main
        main(
            bot=bot,
            color_choice=args.color,
            bot_name=args.bot or "Bot",
            engine=engine,
            motifs=args.motifs,
            analysis_depth=args.analysis_depth,
            analysis_lines=args.analysis_lines,
        )
    else:
        from ui.cli import main
        main(bot=bot, human_color=resolve_color(args.color))


if __name__ == "__main__":
    run()