# Third-party material

The code of this repository is released under the MIT License (see `LICENSE`), with the exceptions below.

## Evaluation tables

The piece-square tables used by the evaluation functions (`bot/evaluation.py`, `bot/advanced_evaluation.py`) are taken from, or adapted from, the
[Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function) page of the Chess Programming Wiki, originally
proposed by Tomasz Michniewski. That page is licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The tables, and any adaptation of them, remain under that license.

## Chess piece images

The images in `ui/assets/pieces/` are the chess piece set by [RhosGFX](https://rhosgfx.itch.io/), released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The license file shipped with the set is kept in `ui/assets/pieces/LICENSE.txt`.

## Lichess puzzle database

The motif classifier in `motifs/` is trained on the [Lichess puzzle database](https://database.lichess.org/#puzzles), released under CC0 1.0.
The database itself is not included in this repository. The pre-trained model in `models/` is derived from it.

## Stockfish

[Stockfish](https://stockfishchess.org/) (GPL v3) is used as an external UCI engine, for rating measurements, blunder analysis and game review.
It is not included in this repository and must be downloaded separately.