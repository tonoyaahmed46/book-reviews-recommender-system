# book-reviews-recommender-system

Recommender system to make book preferences and category prediction based on text-based Goodreads book reviews.

## Dataset descriptions 

- train_Interactions.csv.gz: 200,000 ratings to be used for training
  - userID:  The ID of the user. This is a hashed user identifier from Goodreads
  - bookID: The ID of the book. This is a hashed book identifier from Goodreads
  - rating: The star rating of the user’s review
- train Category.json.gz: Training data for the category prediction task
  - n_votes: The number of ‘likes’ this review has received
  - review_id: A hashed identifier for this review
  - user_id: A hashed identifier for the user
  - review_text:  Text of the review
  - rating_Rating: of the book
  - genreID: A numeric label associated with the genre
  - genre: A string version of the genre
- test_Category.json.gz: Test data associated with the category prediction task.
- pairs_Category.csv: Pairs (userID and reviewID) on which you are to predict the category of a book
- pairs_Read.csv: Pairs on which you are to predict whether a book was read

## Output files descriptions 

- predictions_Read.csv: Given a (user,book) pair from ‘pairs Read.csv’, this file contains predictions of whether the user
would read the book (0 or 1)
- predictions_Category.csv: Predictions of the category of a book from a review. Five categories are used for these predictions: Children’s, Comics/Graphic Novels, Fantasy, Mystery/Thriller, and Romance

## Methodologies in main.py

Part 1: 
- looped over potential thresholds to find the optimal threshold
- created a list of the most popular books using optimal threshold
- used list of the most popular books to create sklearn preprocessor and pipeline  
- used preprocessor and pipeline to predict whether or not book would be read by user in pairs_Read.csv
- stored predictions in predictions_Read.csv

Part 2:
- inputted review training data in a TfidfVectorizer to fit and transform using LinearSVC transformers
- looped through the testing data and extracted all of the reviews
- predicted y valid values using the pipeline 
- matched all of the predicted y valid values with the user IDs and review IDs in the test_Category file
- stored category predictions in predictions_Category.csv
  
