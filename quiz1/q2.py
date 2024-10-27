import numpy as np
import nltk
nltk.download('punkt')

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

# Query string
query = "I would like to order a pizza, but I am not sure what. Something with a lot of cheese, I think!"

# Jaccard similarity function
def jaccard_similarity(doc1, doc2):
    set1 = set(nltk.word_tokenize(doc1))
    set2 = set(nltk.word_tokenize(doc2))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# Compute Jaccard similarity between the query and each document
similarities = [jaccard_similarity(query, doc) for doc in corpus]

# Find the highest similarity
highest_similarity = max(similarities)

# Output the highest similarity
print(f"The highest similarity is: {highest_similarity}")