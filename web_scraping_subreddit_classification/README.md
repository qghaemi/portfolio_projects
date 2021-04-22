# Project 3: Netflix vs. DisneyPlus Subreddit Classification Model

## Problem Statement
Reddit is a social media platform that hosts discussion boards (called Subreddits) on various topics ranging from entertainment, business, politics, and self-help to name a few. Users are able to write posts that other users can interact with by either commenting or "up-voting" posts they like. 

Streaming services, like Netflix and DisneyPlus, are subscription based websites that studios now offer to give viewers direct access to previous and upcoming films & TV shows. As of March 2021, only Sony does not have its own independent streaming service. 

For this project, posts have been scrapped from the Netflix and DisneyPlus Subreddits in order to develop a classification model that will predict which Subreddit the post has been generated from and what is being discussed within these Subreddits.

---

## Executive Summary
To scrape the data required, I used Reddit's API and imported the data in JSON format. The data was pulled from the [r/Netflix](https://www.reddit.com/r/netflix/) and [r/DisneyPlus](https://www.reddit.com/r/DisneyPlus) Subreddit pages. Using Reddit's API data dictionary, I chose to focus on the author, id, upvote_ratio, spoiler, selftext, title, and subreddit features. I chose these Subreddits as opposed to other streaming service's Subreddits because these two have the highest number of subscribers 

    NOTE: Amazon Prime has more subscribers than DisneyPlus, however there is no data that specifies how many Prime users utilize Prime Video so I did not consider it when considering which rankings had the highest number of subscribers.

Next, I used pandas to clean the DataFrame that was collected. I also checked to confirm that the function I created to scrape data did not duplicate any rows. I also created features that counted the length and word count from the title and selftext features. After initial cleaning I began using Natural Language Processing techniques to create more features based on the title and selftext features that had been pulled. 

Once all the cleaning was complete, I began exploring the data to see what trends (if any) can be found to separate the two Subreddits. I chose to check if any features had strong correlation (none had over 50%) and then examined which words came up most frequently. Once I established which words were most prominent I then examined how those words were distributed between the two subreddits. In the below image of the top fifteen words by count, I am able to get a sense of the themes that are most often discussed on both subreddits: new movies or TV shows/seasons/episodes. We can answer the second part of the problem statement with this image and understand that these subreddits are most often used to discuss the newest films or shows.

![Screen%20Shot%202021-03-10%20at%205.12.23%20PM.png](attachment:Screen%20Shot%202021-03-10%20at%205.12.23%20PM.png)

I then examined the word count and length of posts to see how they were distributed between Subreddits. Finally, I examined the sentiment score and up-vote ratio to see if any trends could be found between Subreddits and I found that over 80% of posts with an up-vote ratio below 0.5 and a sentiment score (compounded) above 0 were from r/Netflix (image below for reference).

![sent_up_sub.png](attachment:sent_up_sub.png)

Finally I began modeling. Before I began I decided to focus on creating the model with the highest accuracy score: I wanted to maximize as many correct predictions and ultimately felt that focusing models on this score would provide me with the strongest model. 

I chose three different types of models: Random Forest, K-Nearest Neighbors, and Bernoulli Naive Bayes. I generated a baseline score for each of these models and Random Forest had the highest baseline score so much of my model adjustments came on that model type. I used GridSearchCV, built a Pipeline for feature engineering and cut back on features to only use those with high importance. I was able to build a model that predicted which subreddit a post came from with 84.6% accuracy. 

---

## Conclusion & Recommendations

Based on the word counts collected, we are able to conclude that the themes of each Subreddit is focused on new movies of TV seasons/shows/episodes.

| Model | Training Accuracy Score | Testing Accuracy Score |
| --- | --- | --- |
|Baseline (no model, just picking one Subreddit)|50%|50%|
|Random Forest Baseline|98.3%|82.89%|
|KNN Baseline|81.7%|69.3%|
|Bernoulli Naive Bayes Baseline|78.69%|77.2%|
|Random Forest GridSearchCV|95.69%|84.6%|
|KNN GridSearchCV|98.3%|73.83%|
|Random Forest Feature Engineering & GridSearchCV|94.97%|84.1%|
|Random Forest GridSeachCV: Remove Features w/ Low Importance|94.4%|81.1%|
|Random Forest Feature Engineering & GridSearchCV: Remove Features w/ Low Importance|94.6%|81.5%|

The strongest model (by measurement of accuracy score) was the GridSearchCV Random Forest model with an accuracy score of 84.6%. Given the other Random Forest models that were created all had accuracy scores within 5% of the best score, the GridSearchCV Random Forest model is at or very near the maximum accuracy score with the current data collected. 

Reccomendations for improvement include using other NLP processing tools such as BERT or GloVe. The sentiment score used above was the compounded score but other scores from the sentiment analysis could also be pulled. Another option could be to used GridSearchCV on the Bernoulli Naive Bayes model that showed the least amount of overfitting. A final reccomendation is to collect new data.

Should we choose to used the Random Forest GridSearchCV model we must accept the overfitting but also can sleep well knowing that model will return the highest accuracy score of this batch.