# NLP-tweet-analysis
Code for the "Real or Not? NLP with Disaster Tweets" Competition on Kaggle.

Competition Link: https://www.kaggle.com/c/nlp-getting-started

An NLP-based model to identify tweets pertaining to disasters, natural calamities etc.

I used the Bag-of-Words model to create a corpus of text out of all tweets.
Used Scikit-leran's CountVectorizer to transform the corpora of tweets into a vector of token counts.

For the classifier, I used the Random Forest Classifier with 450 trees, which after much 
trial and error was the model with the best accuracy.


