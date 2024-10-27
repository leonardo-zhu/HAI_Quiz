import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Corpus as provided
corpus = [
    "I'm starving and really fancy a pizza tonight! What kind of pizzas do you have on your menu, and what sizes do they come in? I'm particularly interested in any spicy options you might have.",
    "I'd like to order a large pepperoni pizza for delivery please. Could I also add extra cheese and maybe some jalapeños? What's your delivery area, and do you have a minimum order value?",
    "We're having a family movie night and want to order some pizzas. It's for four people - two adults and two teenagers with big appetites! Are there any meal deals or special offers that would be good value for us?",
    "I'm after a classic Margherita, but with a few extra toppings. Could I get a small Margherita with olives and mushrooms? Also, could you tell me how much that would cost, please?",
    "I have a gluten intolerance, so I need to be careful about what I eat. Do you offer a gluten-free pizza base option? If so, what toppings can I have with that?",
    "My partner and I can never agree on pizza toppings! Is it possible to order a half-and-half pizza with different toppings on each side?",
    "We're pretty hungry, so we might need some sides to go with our pizza. What side dishes do you have available? Do you have any dips or garlic bread?",
    "We need the pizza delivered by 7pm. How long does your delivery usually take? Is there any way to track the order once it's been placed?",
    "I'm vegan, so I'm looking for a pizza with vegan cheese. Do you have any vegan cheese options? What kind of vegan pizzas do you offer?",
    "I'd prefer to pay by card. Can I pay by card on delivery, or do I need to pay online when I order? Do your delivery drivers carry card readers?"
]

# Step 1: Preprocess the data by tokenizing, lowering case, removing stopwords, and stemming
def preprocess(doc):
    tokens = nltk.word_tokenize(doc.lower())  # lowercase and tokenize
    tokens = [word for word in tokens if word.isalnum()]  # keep only alphanumeric tokens
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # stem each token
    return tokens

# Step 2: Build a document-term matrix and count document frequencies (n for each term)
N = len(corpus)  # total number of documents
term_doc_count = defaultdict(int)  # to store how many docs contain each term

# Preprocess each document and count term occurrences
preprocessed_docs = [preprocess(doc) for doc in corpus]
for doc in preprocessed_docs:
    unique_terms = set(doc)  # get unique terms to avoid counting a term more than once per doc
    for term in unique_terms:
        term_doc_count[term] += 1

# Step 3: Calculate IDF for each term using the provided formula
def calculate_idf(term, N, term_doc_count):
    n = term_doc_count[term]
    return 1 + math.log((N / (n + 1)))

# Step 4: Calculate the sum of all IDF values for terms in the corpus
idf_sum = 0
for term in term_doc_count:
    idf_sum += calculate_idf(term, N, term_doc_count)

# Output the sum of all IDF values
print(f"Sum of all IDF values: {idf_sum}")