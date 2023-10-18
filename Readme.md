# HELLO ! And welcome to the Github repo
I should add some documentation to this.
Initially, this was made considering an app called Kite, which is a very easy solution for instrument trading.

The first 160 lines are API and functions that correspond to that particular app which I use to fetch my data.
However you may plug in ANY online stock platform API data, and the code will just work!

Functions:

1. calculate_volatility : uses the standard deviation to calculate amount of volatility (value between 0-1).

   [Also, I have chosen to filter out stocks with prices over 2000 and prices below 5 (penny stocks). It is a personal choice, and it is upto you to decide.]
2. filter_stocks_by_volatility : helper function for above
3. lower_circ_stocks : fetches  all stocks that are below certain percentile.

At the watchlist, you have the option to give specific stock names, take the lowest circuit  stocks or those with highest volatility (or some combination of these).

4. Regression functions : will draw diagonal support-resistance lines for you per stock

   (This is quite difficult to explain, and I had to do some rigorous testing to get it to work.
   But basically, based on the number of maxima and minima, and taking into account other factors like whether the stock is going down or up, it makes the decision to draw the line)

5. adf_test : For performing augmented dickey fuller test
6. create_dataset : to create a X,y dataset to plug into the neural network

I will upload my own results to this repo as time goes on. As of updating this, I have had 7 back to back profits of ~ 1300 - 3400 each.
