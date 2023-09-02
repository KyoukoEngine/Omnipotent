# Omnipotent

QUICK NOTE: my code has a 70% chance of being faulty/unoptimized any issues you find or improvements let me know either in the issues tab
or via discord: SeoTilt



when running the engine-vs-stockfish file edit the src to redirect the stockfish path to your own stockfish path, otherwise it will not work

chess_model is not the actual model, it was to begin with when i was generating the model file, however it is still in use in other files to
use different functions like board_to_array that i am yet to put into another file. 

Benchmark.py is to test the engine with FEN's to pull the best move, not been used since super early stages of this project

This project was started August 31st 2023 (days old) so I am not expecting it to beat stockfish when it plays, its so i can train against 
the strongest opponent in hopes of quicker learning. 

Model.zip contains the training data for the model as well as the model file.

Omnipotent vs stockfish zip contains the pgn files against stockfish on its lowest settings 1 skill 1 depth (yes i am aware my engine isnt strong)

MCTS.py is my w.i.p mcts 

improvement_test.py is used to test new versions of the model against old versions to make sure i didn't break the model and to test to ensure that it is infact learning lol.

