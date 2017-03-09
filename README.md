# Seminar: Finding Patterns in financial stocks

### Requirements

You should install the libraries in the requirement.txt

### Classification Use

To run the backtest, simply run a commande line as follows  in the terminal:

`python classification_algo.py ticker start_date end_date feat_num
 train_period bucket_sampling_period`
 
Example for logistique regression on Yahoo stock:
`python logit_regression.py YHOO 2012-01-02 2016-02-05 30
 120 120`
 
This will only run the classification backtest and print out the accuracy
obtained at the end. There is no strategy or trading according to the signals.
 
### Linear Regression Use 

The current program for linear regression requires the following parameters:

  -regression type (linear, ridge or lasso)
  -alpha of the Ridge or lasso regression (0 if linear)
  -stock name
  -start and end date of the backtest
  -number of features 
  -training period
  -plot_bt ( ‘plt’ in case it’s true, ‘-‘ otherwise) 

For example :
`python linear_regression.py ridge 2 YHOO 2007-01-02 2017-01-10 10 30 plt`

### Neural Network (for regression) Use
The parameters are almost the same:

  -n_layers: Number of hidden layers you want to use 
  -stock name
  -start and end date of the backtest
  -number of features 
  -training period
  -plot_bt ( ‘plt’ in case it’s true, ‘-‘ otherwise) 
  
If you want to use the Neural net:
 `python reg_neural_net.py 1 YHOO 2007-01-02 2017-01-10 10 30 plt`

By running this command line, it will save the historical prices in
the "stocks_hist" folder, generates the backtest and save it in
the "plots_bt" folder.

