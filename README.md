# SE 755 Assignment 1

By Frederick Fogerty // fbru072 // 6653854

## Setup

This package uses dill for model persistence. Dill needs to be installed before running this program. If pip is installed, dill can be installed with: `python -m pip install dill`

## Usage

The command line for this project uses flags to specify what type of data should be processed, and what type of model should be used. 

The CLI can be run with `./cli.py` if python3 is on the PATH and if `cli.py` has the +x permission. Otherwise, `python3 cli.py` can be used.

An explanation of the parameters allowed can be shown using `./cli.py -h`.

A sample output is provided below, but the CLI have been updated since then.
```
usage: cli.py [-h] -t
              {regression,ridge,perceptron,svm,decision_trees,nearest_neighbour,bayes,all}
              -d {wc,traffic,occupancy,landsat,all} [-i INPUT] [-o [OUTPUT]]
              [-s] [-r]

Fit some models.

optional arguments:
  -h, --help            show this help message and exit
  -t {regression,ridge,perceptron,svm,decision_trees,nearest_neighbour,bayes,all}, --type {regression,ridge,perceptron,svm,decision_trees,nearest_neighbour,bayes,all}
                        The type of model to use.
  -d {wc,traffic,occupancy,landsat,all}, --dataset {wc,traffic,occupancy,landsat,all}
                        The dataset the model should be trained on. Can be one
                        of: wc, traffic, occupancy, landsat.
  -i INPUT, --input INPUT
                        File with data to test against.
  -o [OUTPUT], --output [OUTPUT]
                        File to output to. Default is sys.stdout
  -s, --save-model      Save the trained model to disk.
  -r, --re-train        Re-train the model.
```


### Running with an external validation set

To run the analysis with an external validation set, the following command should be used:

```sh
./cli.py -d wc -t regression -i validation_set.csv -o validation_set_pred.csv
```

The predicted results can also be outputted to sys.stdout by omitting the `-o` parameter.

```sh
./cli.py -d wc -t regression -i validation_set.csv
```


**IMPORTANT: It is required that the validation data to be tested have the target columns removed.** For example, testing data for the World Cup dataset should have the `Total_scores` and `Match_result` removed.
