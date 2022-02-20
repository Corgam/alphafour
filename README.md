# Alpha Four Agent
A Connect Four Agent using Alpha Zero Algorithm, made for `Programming Project in Python [Winter Term 21/22]` course at TU Berlin.

### Authors:
- Emil Balitzki
- Sonia Simons
- Douglas Rouse

## Main pipeline
Here we will describe briefly how the main pipeline works. More information can be found in the final presentation.
1. Multiple MCTS self play games are played and all necessary training data is saved on the drive.
2. Training of NN starts using the previously saved data files.
3. Evaluator chooses the better NN, which will be used in the future iteration.
4. If the new NN is not good enough to win with old one, more training data is generated and more training is done, until the new NN is good enough.

## Setup
1. Install all requirements listed in the `requirements.txt` using your favourite package installer (For Pip use: ` pip install -r requirements.txt
`)
2. Run the `main.py` Python script and select the agent you want to play with.
- For running the AlphaFour training type `1`. 
- For Human vs Human match type `2`.
- For a game with an already trained AlphaFour Agent type `3`.
  - Next type the ID of the iteration of the NN to play with. The possible IDs depend on how many NN files are in the `trained_NN` folder.
- For a game with the pure MCTS Agent type `4`.

## Additional Notes
- `common.py` and `helpers.py` files were taken from Emil Balitzki's minmax agent submission (with the possibility of changes present)
- For the docstrings we have used the `reStructuredText` docstrings type, which is automatically supported by the PyCharm (Source: https://www.jetbrains.com/help/pycharm/creating-documentation-comments.html#77db5105).
- All parameters for the main pipeline can be changed manually inside the `main.py` file, above the `main_pipeline` function.
- To make the training fast, the number of iterations, simulations and others are set to small numbers (thus, the agent will be not so smart after default training). As mentioned in previous note, these numbers can be changed, making the training iteration longer, but in the end, the agent smarter.