import os
import requests
import yfinance as yf
import pandas as pd
import openai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from dotenv import load_dotenv

load_dotenv()

newsapi_key = '6e8daddb54064df3ab4acbe28adea707'


openai_api_key = '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'

def collect_news_data(company_name, newsapi_key, start_date, end_date):
    url = 'https://newsapi.org/v2/everything?q={}&from={}&to={}&sortBy=relevancy&apiKey={}'.format(company_name, start_date, end_date, newsapi_key)
    print(url)
    response = requests.get(url)
    news_data = response.json()

    if response.status_code != 200:
        print(f"Error while fetching news data. Status code: {response.status_code}")
        print(f"Response: {news_data}")
        return None

    return news_data




def extract_article_info(news_data):
    articles = []
    for article in news_data['articles']:
        title = article['title']
        description = article['description']
        published_at = article['publishedAt']
        articles.append({
            'title': title,
            'description': description,
            'publishedAt': published_at
        })
    return articles

def collect_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

def generate_summary(news_articles, openai_api_key, company_name):
    openai.api_key = openai_api_key
    prompt = "Create a summary for the following news articles about {}:\n".format(company_name)
    
    for article in news_articles:
        prompt += "- {}\n".format(article["description"])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that accurately summarizes news articles very briefly - each summary representing the sentiment of each article accurately  (< 75 words)"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=70 * len(news_articles),
        n=1,
        stop=None,
        temperature=0.76
    )

    summary = response['choices'][0]['message']['content'].strip()
    return summary.split('\n')

    
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)["compound"]
    return sentiment


def combine_data(news_articles, stock_data):
    print("News articles in combine_data():")
    for article in news_articles:
        print(article)
    combined_data = stock_data.copy()
    combined_data['Date'] = pd.to_datetime(combined_data['Date']).dt.date

    news_dates, news_summaries, news_sentiments = [], [], []

    for article in news_articles:
        if 'publishedAt' in article and 'summary' in article and 'sentiment' in article:
            news_dates.append(article['publishedAt'])
            news_summaries.append(article['summary'])
            news_sentiments.append(article['sentiment'])

    news_df = pd.DataFrame({'Date': news_dates, 'Summary': news_summaries, 'Sentiment': news_sentiments})
    combined_data = combined_data.merge(news_df, on='Date', how='outer')

    return combined_data


def analyze_data(combined_data):
    avg_sentiment = combined_data[combined_data['Sentiment'].notnull()]['Sentiment'].mean()
    positive_days = len(combined_data[combined_data['Sentiment'] > 0])
    negative_days = len(combined_data[combined_data['Sentiment'] < 0])

    insights = {
        'average_sentiment': avg_sentiment,
        'positive_days': positive_days,
        'negative_days': negative_days
    }

    return insights
def rolling_sentiment(combined_data, window=7):
    combined_data["RollingSentiment"] = combined_data["Sentiment"].rolling(window=window, min_periods=1).mean()
    return combined_data

def plot_corr_heatmap(combined_data):
    corr_matrix = combined_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
def plot_data(combined_data):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price')
    ax1.plot(combined_data['Date'], combined_data['Close'], label='Stock Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment')
    ax2.scatter(combined_data['Date'], combined_data['Sentiment'], label='Sentiment', color='tab:red', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.show()

def generate_insights(insights, openai_api_key, company_name):
    openai.api_key = openai_api_key
    prompt = "Based on the average sentiment of {}, with {} positive days and {} negative days, provide insights and recommendations for investors and analysts about {}.".format(insights['average_sentiment'], insights['positive_days'], insights['negative_days'], company_name)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides insights and recommendations based on sentiment analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    insights = response['choices'][0]['message']['content'].strip()
    return insights


def bulk_generate_summaries(news_articles, openai_api_key, company_name, batch_size=5):
    tasks = []
    for i in range(0, len(news_articles), batch_size):
        batch = news_articles[i:i+batch_size]
        summaries = generate_summary(batch, openai_api_key, company_name)
        tasks.append(summaries)
    
    summaries = [summary for batch in tasks for summary in batch]
    return summaries


def main():
    company_name = input("Enter the company name: ")
    company_ticker = input("Enter the company ticker: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    news_data = collect_news_data(company_name, newsapi_key, start_date, end_date)

    if news_data is None:
        print("Unable to fetch news data. Please check your API key and try again.")
        return

    stock_data = collect_stock_data(company_ticker, start_date, end_date)

    news_articles = extract_article_info(news_data)

    summaries = bulk_generate_summaries(news_articles, openai_api_key, company_name)
    for article, summary in zip(news_articles, summaries):
        article['summary'] = summary
        article['sentiment'] = analyze_sentiment(summary)
    print("News articles with summaries and sentiment:")
    for article in news_articles:
        print(article)


    combined_data = combine_data(news_articles, stock_data)
    combined_data = rolling_sentiment(combined_data)
    insights = analyze_data(combined_data)
    plot_data(combined_data)
    plot_corr_heatmap(combined_data)
    generated_insights = generate_insights(insights, openai_api_key, company_name)

    print("\nGenerated Insights:")
    print(generated_insights)

if __name__ == '__main__':
    main()
