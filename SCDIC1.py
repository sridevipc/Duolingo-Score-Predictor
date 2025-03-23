import requests
import json
import nltk
import csv
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure required NLTK data is available
nltk.download('wordnet')

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to scrape COVID-19 glossary terms from CDC & WHO
def get_covid_terms():
    urls = [
        "https://www.cdc.gov/coronavirus/2019-ncov/your-health/about-covid-19/basics-covid-19.html",
        "https://www.who.int/health-topics/coronavirus#tab=tab_1"
    ]

    terms = set()

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract terms based on common structures
        for li in soup.find_all("li"):
            text = li.get_text().strip().lower()
            if 1 < len(text.split()) < 5:  # Simple filter for short medical terms
                terms.add(text)

    return list(terms)


# Function to get sentiment polarity using VADER
def get_sentiment_polarity(word):
    score = analyzer.polarity_scores(word)["compound"]
    return round(score, 3)


# Function to get semantic relationships using WordNet
def get_semantics(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))

    return ", ".join(list(synonyms)[:5]) if synonyms else "N/A"


# Function to determine primary and secondary emotion
def get_emotions(polarity):
    if polarity > 0:
        return "#contentment", "#acceptance"
    elif polarity < 0:
        return "#grief", "#loathing"
    else:
        return "#neutral", "None"


# Function to calculate introspection, temper, attitude, sensitivity
def compute_emotional_scores(polarity):
    introspection = round(polarity * 0.8, 3)
    temper = round(polarity * 0.6, 3)
    attitude = round(polarity * 0.5, 3)
    sensitivity = round(polarity * 0.4, 3)

    return introspection, temper, attitude, sensitivity


# Main function to generate COVID-19 SenticNet dictionary
def generate_covid_senticnet():
    covid_terms = get_covid_terms()
    covid_sentic_dict = []

    for term in covid_terms:
        polarity = get_sentiment_polarity(term)
        introspection, temper, attitude, sensitivity = compute_emotional_scores(polarity)
        primary_emotion, secondary_emotion = get_emotions(polarity)
        semantics = get_semantics(term)

        covid_sentic_dict.append({
            "CONCEPT": term,
            "INTROSPECTION": introspection,
            "TEMPER": temper,
            "ATTITUDE": attitude,
            "SENSITIVITY": sensitivity,
            "PRIMARY EMOTION": primary_emotion,
            "SECONDARY EMOTION": secondary_emotion,
            "POLARITY VALUE": polarity,
            "POLARITY INTENSITY": abs(polarity),
            "SEMANTICS": semantics
        })

    # Save as CSV file
    with open("covid_senticnet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=covid_sentic_dict[0].keys())
        writer.writeheader()
        writer.writerows(covid_sentic_dict)

    print("COVID SenticNet dictionary saved as covid_senticnet.csv")


# Run the script
generate_covid_senticnet()
