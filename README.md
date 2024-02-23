# Spam or Ham Classification with Text Mining

This README details a project focused on classifying messages as spam or ham (not spam) using various text mining techniques and machine learning models. The project involves preprocessing the text data, feature engineering, and applying classification algorithms to determine the effectiveness of each method.

## Project Overview

- **Objective**: To create a classifier that can accurately identify spam and ham messages.
- **Data**: The dataset contains messages labeled as spam or ham.
- **Methods**: The project utilizes Lasso, Ridge, and ElasticNet regression methods to classify messages.
- **Text Mining**: Involves preparing the text for analysis (e.g., removing unnecessary characters, converting to lowercase) and then using the prepared text to predict message classifications.

## Steps in the Project

1. **Load and Preprocess Data**: Read the dataset and rename columns for clarity.
2. **Feature Engineering**:
    - Count the number of words and characters in each message.
    - Convert all messages to lowercase.
    - Use Regular Expressions to count and remove symbols.
    - Download and count stopwords, then remove them from each message.
    - Apply stemming to reduce words to their base form.
3. **Data Analysis**:
    - Compare features (word count, character count, etc.) between spam and ham messages.
    - Create a Bag of Words (BOW) matrix to transform text into numerical features.
4. **Model Training and Evaluation**:
    - Split the data into training and testing sets.
    - Train Lasso, Ridge, and ElasticNet models on the training data.
    - Evaluate each model's accuracy on the test set, specifically looking at the classification of spam and ham messages.

## Results

- The accuracy of each model (Lasso, Ridge, ElasticNet) is compared to determine which provides the best performance in classifying messages as spam or ham.
- The project also explores the frequency of words in spam and ham messages, removing words that appear less than 20 times to focus on more significant terms.

## Conclusion

This project demonstrates the application of text mining and machine learning techniques to classify messages as spam or ham. By preprocessing the text data, engineering relevant features, and applying different regression models, we can assess which method provides the most accurate message classification. The comparison of model accuracies helps in identifying the most effective approach for spam detection.
