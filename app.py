from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import feedparser
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from textblob import TextBlob
import base64
from io import BytesIO
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    source = request.form['source']
    return redirect(url_for('display_news', source=source))

@app.route('/display_news/<source>')
def display_news(source):
    source_map = {
        'The Indian Express': 'https://news.google.com/rss/search?q=source:"The+Indian+Express"',
        'Times of India': 'https://news.google.com/rss/search?q=source:"The+Times+of+India"'
    }
    feed_url = source_map.get(source, '')
    if not feed_url:
        return "News source URL not found", 404

    feed = feedparser.parse(feed_url)
    articles = feed.entries[:35]  # Get top 10 news articles

    return render_template('display_news.html', articles=articles, source=source)


@app.route('/analyze')
def analyze():
    url = request.args.get('url')
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Attempt to determine the source by checking meta tags or known content structures
    meta_publisher = soup.find("meta", property="og:site_name")
    publisher = meta_publisher["content"].lower() if meta_publisher else ""

    # Set selectors based on the publisher identified by meta tags
    if 'times of india' in publisher:
        content_blocks = soup.find_all('div', class_="_s30J clearfix")
    elif 'indian express' in publisher:
        content_blocks = soup.find_all('div', class_="story_details")
    else:
        return render_template('error.html', message="Unsupported news source.")

    article_content = ' '.join(block.text.strip() for block in content_blocks if block.text.strip() != '')

    if not article_content:
        return render_template('error.html', message="No content found to analyze.")

    try:
        vectorizer = CountVectorizer(stop_words='english')
        bow_matrix = vectorizer.fit_transform([article_content])
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

        bow_data = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray()[0]))
        tfidf_data = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

        top_bow_words = sorted(bow_data.items(), key=lambda item: item[1], reverse=True)[:3]
        top_tfidf_words = sorted(tfidf_data.items(), key=lambda item: item[1], reverse=True)[:3]

        # Generate synthetic data for a more robust demonstration
        np.random.seed(42)
        synthetic_data = np.random.normal(0, 0.01, (10, tfidf_matrix.shape[1])) + tfidf_matrix.toarray()
        synthetic_y = np.random.rand(10) * 100  # Random target values for demonstration

        combined_data = np.vstack([tfidf_matrix.toarray(), synthetic_data])
        combined_y = np.hstack([np.mean(synthetic_y), synthetic_y])

        model = Ridge(alpha=1.0)
        model.fit(combined_data, combined_y)
        predictions = model.predict(combined_data)

        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(combined_y)), combined_y, color='blue', label='Actual Values')
        plt.plot(range(len(predictions)), predictions, color='red', label='Predicted Values')
        for i, txt in enumerate(range(len(combined_y))):
            plt.annotate(txt, (i, combined_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('Ridge Regression Analysis')
        plt.xlabel('Data Points')
        plt.ylabel('Values')
        plt.legend(loc='best')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Perform sentiment analysis
        blob = TextBlob(article_content)
        sentiment_score = blob.sentiment.polarity
        sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        return render_template('analysis_news.html', article_content=article_content, bow_data=bow_data, tfidf_data=tfidf_data, image_data=image_base64, top_bow_words=top_bow_words, top_tfidf_words=top_tfidf_words, sentiment_score=sentiment_score, sentiment_label=sentiment_label)
    except Exception as e:
        return render_template('error.html', message=str(e))
    #     return render_template('analysis_news.html', article_content=article_content, bow_data=bow_data, tfidf_data=tfidf_data, image_data=image_base64, top_bow_words=top_bow_words, top_tfidf_words=top_tfidf_words)
    # except Exception as e:
    #     return render_template('error.html', message=str(e))
# @app.route('/ridge_graph')
# def ridge_graph():
#     # Add code to generate the Ridge Regression graph
#     # Return the template for displaying the Ridge Regression graph
#     return render_template('ridge_graph.html')

# @app.route('/sentiment_analysis')
# def sentiment_analysis():
#     # Add code to perform sentiment analysis
#     # Return the template for displaying the sentiment analysis
#     return render_template('sentiment_analysis.html')

if __name__ == "__main__":
    app.run(debug=True)
