Norman Hong  
  
Introduction:  
  
Reddit comment data from October 2018 to Jan 2019 was used to develope a predictive model in which to identify if a comment made is controversial or not.  Comments are controversial when there is a high number of up and down votes.  For instance, a comment with 100 upvotes and 100 downvotes is considered controversial.  Because the size of the data set is 500 gb, pyspark was used.  Pyspark is able to handle semi-structured data very well and does this faster than pig.  

Code:  
  
The code with the model is contained in the Untitled.ipynb file.  This is a jupyter notebook file that details the step-by-step analysis.  

Method:  

In order to filter out some of the noise in the data set, only subreddits where the topic is generally controversial  are considered.  Using domain knowledge, "politics", "The_Donald", "nfl", and "pics" were chosen.  Several columns contained a lot of missing values.  These columns were dropped from the analysis.  Also, random identifiers were also dropped because these variables contain no information of interest.  A "hour" feature was created to determine the hour at which the comment was posted.  A "DayofWeek" variable was created to determine which day the comment was posted.  Lastly, a "comment_length" variable was made to track the number of characters in each comment.  Several binary variables had to be transformed into a dummy variable with 1 and 0 before being fed into the algorithm.  Because the problem of interest is a binary classification problem, a logistic regression model and a random forest model were used.  The auc measure was used to evaluate the model.  

Results:

The random forest and logistic regression models both have auc scores of 0.5.  This means that the models are no better than randomly guessing the controversiality of each comment.  The only way around this is to collect more data on different variables or create new features from old ones.

Future work:

In corporating NLP techniques to analyze the text of the comment might lead to better results.  Therefore, the next step is to analyze the actual body of the comment.  

