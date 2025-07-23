import bs4 as bs
import urllib.request
import re
import nltk
import heapq
nltk.download('punkt_tab')

url = 'https://www.nltk.org/book/ch01.html'

'''Importing the data Using the most common webscraping format, we import the data based off of a url and pick up the 
paragraphs. I have tested this for limited sites, and if crucial information is in headdings or other htypes of tags, 
it will get missed. This did not have to get built into a function, but I would like to keep this handy for 
reusablility.'''


def get_data(url):
    # PRE: beautiful soup and nltk are imported
    scraped = urllib.request.urlopen(url).read()
    parsed = bs.BeautifulSoup(scraped, 'lxml')
    paragraphs = parsed.find_all('p')
    text = ""
    for p in paragraphs:
        text += p.text
    # tokens = nltk.sent_tokenize(text)
    return text


'''Preprocessing
Much like any Data science project, just because we have the data, does not mean that it does not require any further refinement. With the scraped data we hope to reduce the text to readable messages, tokenize the data, and rule out stopwords.
'''

data = get_data(url)

data = re.sub(r'\[[0-9]*\]', ' ', data)
data = re.sub(r'\s+', ' ', data)
data = re.sub('[^a-zA-Z]', ' ', data)
data = re.sub(r'\s+', ' ', data)

tokens = nltk.sent_tokenize(data)



'''
Instead of brainstorming my own set of stopwords, like I have in the past, I am simply poaching the nltk coprus collection. This is a list of words that is far more elaborate than what I could come up with and is a hundred times easier to develop. This list consists of 179 words that are all too common in speech to see recognize a signigificant use.

The reason that eliminating stopwords is so important in this project is that our stratagy is dependant on the word frequency in each sentance, and using common words would cause an unnecessary noise that the model would not be able to see through.
'''

stopwords = nltk.corpus.stopwords.words('english')


'''
Word Frequency: Below we fufil our strategy of getting the word frequency of each word in the sentances. We include the word frequncy because theoretically, if a word is used more frequently, the sentance it pertains to is more important. Thus we want to iterate through the article to find the most freqently used words.
'''
def get_word_frequency(formatted_article):
    word_frequency = {}
    for word in nltk.word_tokenize(formatted_article):
        if word not in stopwords:
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1
    return word_frequency

word_frequncy = get_word_frequency(data)


'''Sentance Score
Similar to above, we need to iterate through the entire text. Utilizing the frequncy from above, we sum them into the score below and get the relevance score of all of the data.
'''
def get_sentance_score(tokenize_sentance, freq):
    score = {}
    for sent in tokenize_sentance:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freq.keys():
                if len(sent.split(' ')) < 50000:#ORgially 50
                    if sent not in score.keys():
                        score[sent] = freq[word]
                    else:
                        score[sent] += freq[word]
    return score

word_frequncy = get_word_frequency(data)
sentance_score = get_sentance_score(tokens, word_frequncy)



'''
Output the sparknotes
Finally, we have built the scores for getting the most important information. Now using the heapq package we can output them in a priority queue with the most important data.
'''

sentence_summary = heapq.nlargest(30, sentance_score, key = sentance_score.get)
summary = ' '.join(sentence_summary)
print(summary)
