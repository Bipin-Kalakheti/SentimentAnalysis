import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')  # Set the visual style of plots to 'ggplot'

import nltk

# Ensure that the VADER lexicon is available
nltk.download('vader_lexicon')

# Load data from a CSV file into a DataFrame
df = pd.read_csv('Reviews.csv')
print(df.shape)  # Print the shape of the DataFrame
df = df.head(500)  # Select the first 500 rows
print(df.shape)  # Print the new shape of the DataFrame

df.head()  # Display the first few rows of the DataFrame

# Create a bar plot showing the count of reviews by star rating
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')  # Set the label for the x-axis
plt.show()  # Show the plot

# Extract an example text from the DataFrame
example = df['Text'][50]
print(example)  # Print the example text

# Tokenize the text
tokens = nltk.word_tokenize(example)
tokens[:10]  # Display the first 10 tokens

# Part-of-speech tagging
tagged = nltk.pos_tag(tokens)
tagged[:10]  # Display the first 10 tagged tokens

# Named entity recognition
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()  # Pretty-print the named entities

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Get polarity scores for example sentences
sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)  # Get polarity scores for the extracted example

# Compute sentiment scores for all texts in the DataFrame
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):  # Progress bar
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Convert the results dictionary to DataFrame and merge with the original data
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

vaders.head()  # Display the first few rows of the combined DataFrame

# Create a bar plot showing compound sentiment score by Amazon star review
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

# Create subplots for positive, neutral, and negative sentiment scores
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize tokenizer and model for sentiment analysis
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Encode text for processing by the model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)  # Print the sentiment scores from the model

# Define a function to compute sentiment scores using Roberta model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict  # Return the sentiment scores

# Compute sentiment scores for all texts using both VADER and Roberta
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')  # Handle exceptions during processing

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

from transformers import pipeline

# Initialize a sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

# Test the pipeline with sample sentences
sent_pipeline('I love sentiment analysis!')
sent_pipeline('Make sure to like and subscribe!')
sent_pipeline('booo')

# Create a pairplot to visualize the relationship between different sentiment scores
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score', palette='tab10')
plt.show()  # Display the pairplot
