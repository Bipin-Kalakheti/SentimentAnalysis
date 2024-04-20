# Sentiment-Analysis-ML-Flask-App-Created-by-group6

A machine learning end-to-end Flask web app for sentiment analysis, created using Scikit-learn & VADER Sentiment. This application processes text inputs to determine sentiment scores, illustrating how natural language processing can be applied to extract meaningful insights from text data.

## Project Dependencies

This project uses the following libraries:

- **Flask**: A lightweight WSGI web application framework.
- **Sklearn**: A machine learning library for Python.
- **Requests**: A library for making HTTP requests in Python.
- **NLTK**: A leading platform for building Python programs to work with human language data.
- **RE**: Python's built-in library for regular expressions.
- **vaderSentiment**: A lexicon and rule-based sentiment analysis tool that is particularly good at handling social media text and similarly constructed data.

## VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains.

Learn more about VADER: [VADER on PyPI](https://pypi.org/project/vaderSentiment/)

## About Sentiment Analysis

Sentiment analysis is a critical area of Natural Language Processing that involves analyzing opinions or feelings expressed in textual data. Widely applied to customer feedback like product reviews, sentiment analysis helps businesses understand public sentiment which can influence decision making.

Use cases extend across analyzing sentiments on:

- Product reviews on e-commerce platforms.
- Public opinions on social media platforms.
- Feedback on movie and product services.

## Application Overview

The application provides a web interface where users can input text. The text is then processed to remove noise such as punctuation and stopwords. Using the VADER Sentiment tool, the application analyzes the text to determine its sentiment. The sentiment scores are displayed in the web interface, providing insights into the overall sentiment and specific emotional scores like positivity, negativity, and neutrality.

## Screenshot

Here's what the output looks like:
![Sentiment Analysis Result](sentiment.gif)

## Running the Application

To run the application locally, ensure you have Python installed, then follow these steps:

1. Clone the repository:
   git clone https://github.com/yourusername/Sentiment-Analysis-ML-Flask-App.git

2. Navigate to the cloned directory:
   cd Sentiment-Analysis-ML-Flask-App

3. Install the required dependencies:
   pip install -r requirements.txt

4. Start the Flask server:
   python app.py

5. Open a web browser and go to `http://127.0.0.1:5002` to use the application.
