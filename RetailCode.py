# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:23:47 2019

@author: daniel.lopez
"""

# IMPORT PACKAGES
import pandas as pd
import numpy as np
import collections
from collections import Counter
import itertools
import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import random
import re
import json

# CREATE RECIPES DATASET

# import recipes. (https://www.kaggle.com/hugodarwood/epirecipes)
with open('C:/Users/daniel.lopez/Documents/GitHub/retail/full_format_recipes.json') as json_data:
    new_recipes = json.load(json_data)
    json_data.close()  

# Create a Dataframe with the recipes 
keys = list(new_recipes[0].keys()) # get important keys
df = pd.DataFrame.from_dict(new_recipes, orient='columns')
df['Key'] = range(1, len(df)+1,1)
df['Terms'] = df['ingredients']

# create directions and recipes lists
recipe_list = list()
directions_list = list()

# Create individual lists for ingredients.
for i in range(0,len(new_recipes)):
    try:
        recipe = list(new_recipes[i]['ingredients'])
        words_to_eliminate = ['mg','additional','accompaniments','accompaniment','about','a ','ml','oz','ounce','ounces','cups','cup','spoon','spoons','teaspoon','teaspoons','tablespoon','tablespoons','pound','pounds','lb','pinch','Pinch','inch'] # words i don't need I just want the ingredients
        remove = '|'.join(words_to_eliminate)
        recipe = [re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE).sub("",w) for w in recipe]
        recipe = [re.sub(r'\([^)]*\)', "", w) for w in recipe]
        recipe = [re.sub(r'[\[\],\')Â¿?^*"(\.;$%_–:+<>⁄•—'',%§––¾½¼]', "", w) for w in recipe]
        recipe = [re.sub(r'[0-9/,-]', "", w) for w in recipe]
        recipe = [re.sub(r'^\s+', "", w) for w in recipe]
        recipe = [re.sub(r'\s+', " ", w) for w in recipe]
        recipe = [x.lower() for x in recipe]
        recipe = list(filter(None, recipe))
        recipe_list.append(recipe)
        directions_list.append(new_recipes[i]['directions'])
    except:
        pass
    
"""
Here what we do is first remove all the measurement words that appear in the recipes
dataset since we are only concerned about the individual products we will use, this is
the ingredients that appear in those recipes. Then we remove all special characters 
and transform variables into lower case to assure homogenity
"""

# Obtain the unique ingredients 
list_prods_in_recipes = list()
for recipe in recipe_list:
    list_prods_in_recipes += recipe
    
"""
We separete all the ingredients in the list into individual items
"""
 
# Remove duplicate values from the list   
list_prods_in_recipes = list(set(list_prods_in_recipes)) # set deletes duplicates

# Remove any empty values that can be generated 
list_prods_in_recipes = [x for x in list_prods_in_recipes if len(x.strip())]

# Individual ingredients in our dataset
display(len(list_prods_in_recipes))

# Load products sold by the retailer
products = pd.read_csv("C:/Users/daniel.lopez/Documents/GitHub/retail/products.csv")

# Transoform variables into lower case

products['product_name'] = products['product_name'].str.lower()

# Remove all special characters
products["product_name"] = products["product_name"].str.replace(r"[^a-zA-Z ]+", " ").str.strip()

products["product_name"] = products["product_name"].str.replace(r'[0-9/,-]+', " ").str.strip()

products["product_name"] = products["product_name"].str.replace(r'[\[\],\')Â¿?^*"(–\.;$%_:+<>⁄•—,%§‟––¾½¼]', " ").str.strip()

# Put into a list the produts - these products are the ones sold by the retailer 

sold_products_list = list(products['product_name'])
     
# Now that I have the individual ingredients I remove the duplicate values

sold_products_list = list(set(sold_products_list)) 

# Remove any empty values that can be generated 
sold_products_list = [x for x in sold_products_list if len(x.strip())]

# Create list of terms that is going to be used to train the word2vec model. 

combined_list = sold_products_list + list_prods_in_recipes
"""
We combine both of the lists into an individual list because we have to train
the model with all of the products we have at hand. Products that do not appear
in this list will not be recognised by the model.
"""

display('Number of products sold by the retailer: {}'.format(len(sold_products_list)))
display('Number of ingredients from the recipes dataset: {}'.format(len(list_prods_in_recipes)))
display('Total number of products sold + ingredients from the recipes dataset: {}'.format(len(combined_list)))

#Create a table with the products of both lists
product_table = pd.DataFrame(combined_list, columns=['Products'])

# Separate all words with a ' ' space
product_table['prod_words']  = product_table['Products'].str.replace('\W', ' ')

# Split products into terms: Tokenize.
product_table['prod_words'] = product_table['prod_words'].str.split()
"""
We separate words and then tokenise them to get the individual word which 
will be the ingredient
"""

# Define the function for word2vec

def term_vectors(list_terms):

    """
    Returns a dictionary with term and its vector. 

    Parameters:
    argument1 (list): List of products in tokenized form. ex: [[chocolate, sandwich, cookies],[...], ...]

    Returns:
    dict:Vectors

    """
    
    w2vec_model = Word2Vec(list_terms, size=20, window=5, min_count=1, workers=4)

    prod_word = dict()
    for w in w2vec_model.wv.vocab:
        prod_word[w] = w2vec_model[w]
        
    w2vec_model.save("word2vec_ingredients_products.model")  
    
    return prod_word

#w2vec_model = Word2Vec(list(product_table['prod_words']), size=20, window=5, min_count=1, workers=4)

#Generate vectors based on the whole product and ingredient list we have 
    
word_vectors = term_vectors(list(product_table['prod_words']))

"""
Generates a vector for each word that there is in all the products.
"""

# Calculate the average for the vectors produced in the previous step

def doc_vectors(word_dict,  doc_df):

    """
    Returns a dictionary with document and its vector. 

    Parameters:
    argument1 (list): List of products in tokenized form. ex: [[chocolate, sandwich, cookies],[...], ...]

    Returns:
    dict:Vectors

    """
    
    doc_w2vec = dict()
    
    for doc in doc_df.iterrows():
        
        doc_vector = list()
        for term in doc[1]['Terms']:
            doc_vector.append(word_dict[term])

        doc_w2vec[doc[1]['Key']] = np.average(doc_vector, axis=0)
    
    return doc_w2vec
"""
Returns a dictionary with the products and the average vector based on the 
words used in that product.
"""         

# Generate product vectors.
# Rename the columns of the product table 
product_table = product_table.rename(index=str, columns={"Products": "Key", "prod_words": "Terms"})

# Generate a dictionary with the average vector of all the products
product_vectors = doc_vectors(word_vectors, product_table)  
"""
This generates a dictionary with all the average vectors for all the products 
there are available, in both of the recipes and products sold by the vendor
"""

# create recipes dataframe
recipes_df = pd.DataFrame({'Key':range(1, len(recipe_list)+1,1), 'ingredients':recipe_list,
                                         'directions':directions_list}) 
    
recipes_df = recipes_df.rename(index=str, columns={"ingredients": "Terms"})

# Remove empty terms

recipes_df = recipes_df[recipes_df.astype(str)['Terms'] != '[]']

#
recipe_vectors = doc_vectors(product_vectors, recipes_df)
"""
For each recipe we generate a vector based on the vectors of the products sold
"""

# Play with the model

model = gensim.models.Word2Vec.load("word2vec_ingredients_products.model")

# Find the most similar terms 
model.wv.most_similar('oil',topn=5)

# Similarity between 2 terms

model.wv.similarity(w1='pasta',w2='penne')

####################################################################################
#                               Plot                                               #
####################################################################################

keys = ['pasta','water','oil','chicken','beer','tomato','garlic']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.wv.most_similar(word, topn=10):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

import matplotlib.cm as cm


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

####################################################################################

# Creating the annoy 

# Number of dimensions of the vector annoy is going to store. 
# Make sure it's the same as the word2vec we're using!
f = 20

# Specify the metric to be used for computing distances. 
u = AnnoyIndex(f, metric='angular') 

# We can sequentially add items.
for key,val in recipe_vectors.items():
    u.add_item(key, val)

# Number of trees for queries. When making a query the more trees the easier it is to go down the right path. 
u.build(10) # 10 trees


# Try the annoy for the product vectors 

# Number of dimensions of the vector annoy is going to store. 
# Make sure it's the same as the word2vec we're using!
f = 20

# Specify the metric to be used for computing distances. 
w = AnnoyIndex(f, metric='angular') 

# We can sequentially add items.
for key,val in enumerate(product_vectors.values()):
    w.add_item(key, val)

# Number of trees for queries. When making a query the more trees the easier it is to go down the right path. 
w.build(10) # 10 trees

#Recipe recommendation 
# Let's look for values in a given term

products[products['product_name'].str.contains("oil")]

# Create a ficticious cart 

cart = ['bacon', 'pasta', "tomato"]

#cart_vector = doclist_2_vector(product_vectors, cart)

def basket_average(product_vectors, cart):
    basket = list()
    for i in cart:
        
        basket.append(product_vectors[i])
    vec_avg = np.average(basket, axis=0)
    return vec_avg
      
cart_vector = basket_average(product_vectors, cart)

print(cart)
print(cart_vector)

#similar_products = u.get_nns_by_item(10, 10)
similar_recipes = u.get_nns_by_vector(cart_vector, 5, search_k=-1, include_distances=False)
print(similar_recipes)

# Find recipes

#display(similar_baskets)
for b in similar_recipes:
    print(b)
#    print(recipes_df[df.Key==b].calories)
#    print(recipes_df[df.Key==b].rating)
    for ingredient in list(recipes_df[recipes_df.Key==b].Terms):
        #w.similar_by_vector(vector, 5, search_k=-1, include_distances=False)
        print(ingredient)
        

###############################################################################
        # Create a recommender 
###############################################################################

# Load the orders         
orders_df = pd.read_csv("C:/Users/daniel.lopez/Documents/GitHub/retail/orders.csv") 

# Visualise the graph
orders_df['user_id'].value_counts().hist(bins=100)

#Calculate the mean
print(np.mean(orders_df['user_id'].value_counts()))
print(np.median(orders_df['user_id'].value_counts()))
"""
The mean number of purchases is 17 and the median is 10
"""

# Visualise 
orders_df['days_since_prior_order'].hist(bins=100)

# Average orders
print(np.mean(orders_df['days_since_prior_order']))
"""
2 to 3 orders per month
"""
 
# count group orders
orders_count = orders_df['user_id'].value_counts().to_frame().reset_index().rename(index=str, columns={'user_id':'count'})
print(orders_count[orders_count['count']>10]['index'].count())
"""
We group customers by id and count the number of transactions.
We have a set of 206.209 unique customers, out of which nearly 50% have purchased
more than 10 times.
"""

# Best customers

orders_df_masked = orders_df[orders_df['user_id'].isin(orders_count[(orders_count['count']>10) & (orders_count['count']<15)]['index'])]
print(np.mean(orders_df_masked['days_since_prior_order']))
"""
We define oir best customers as those who have purchased more than 10 times and less 
than 15 since these are the ones with the highest growth potential
"""

print('Number of original orders: {:,}'.format(len(orders_df_masked.order_id.unique())))
print('Number of original orders: {:,}'.format(len(orders_df.order_id.unique())))

# Masked orders combine

orders_df_masked = orders_df_masked.groupby('user_id')['order_id'].apply(list)
orders_df_masked = orders_df_masked.reset_index()#.head()
orders_df_masked.head()
"""
We group all these best customers and group them by their order_id's
"""

# Obtain all unique products
order_ids = list()
for order in orders_df_masked.order_id:
    order_ids +=order
"""
Obtain all the code for the unique id's that the customers have purchased
"""    

# Load the order id table where we have the id's of the products ordered
order_products_prior = pd.read_csv("C:/Users/daniel.lopez/Documents/GitHub/retail/order_products__prior.csv")

# Baskets 
baskets = order_products_prior[order_products_prior['order_id'].isin(order_ids)]
"""
Obtain, from the order id's previously analysed all the details of that order,
product id's etc.
"""

#Group order id's and product's ordered
baskets_df_masked = baskets.groupby('order_id')['product_id'].apply(list)
"""
Now we have all the "tickets" all the products that were purchased together
"""

# Create dictionary
product_id_pair=dict()
for product in products.iterrows():
    product_id_pair[product[1].product_id] = product[1].product_name
"""
For all the product id's that we have in our lists, return the product name
from the products table
"""

# create the basket vector    

basket_vectors = dict()
for basket in baskets_df_masked.iteritems(): 
    vector_list = list()
    
    for product in basket[1]:
        try:
            if  product > 0:
                vector_list.append(product_vectors[product_id_pair[product]])
        except: print('you shall not pass')
        
    basket_vectors[basket[0]]=np.average(vector_list, axis=0)  
"""
Based on the products in each basket, generate the vector for that products 
ordered.
"""
        
# User vector 
#user_vectors = dict()
#list_of_prods_analysed=list()
#userid_list = list()
#for user in orders_df_masked.iterrows(): 
#    vector_list = list()
#    for basket in user[1].order_id:# check what each customer has ordered
#        try:
#            vector_list.append(basket_vectors[basket])# for all the products ordered bring the vector generated in the dict
#            list_of_prods_analysed.append(basket)
#            userid_list.append(user[1].user_id)
#            put = np.average(vector_list, axis=0) 
#            user_vectors.update(user[1].user_id = put)
#        except:
#            pass

# User vector 
user_vectors = dict()
for user in orders_df_masked.iterrows(): 
    vector_list = list()
    #display(user[1])
    for basket in user[1].order_id:
        #display(basket_vectors[basket])
        try:
            vector_list.append(basket_vectors[basket])
        except:
            pass
#    user_vectors[user[0]]=np.average(vector_list, axis=0)
    user_vectors[user[1].user_id] = np.average(vector_list, axis=0)        

"""
Here we check what the customers have ordered in every order by product id.
We check these product id's against the dictionary and generate a vector 
of the average values of the vector
"""
    
print(list(user_vectors.items())[:3])            

# Recommender

similar_recipes = u.get_nns_by_vector(user_vectors[1], 5, search_k=-1, include_distances=False)

# looking for list of directions in recipes dataframe

list_directions_trial = list()
list_ingredients_trial= list()
list_user_id_trial = list()

for b in similar_recipes:
    list_directions_trial.append(recipes_df[recipes_df.Key==b].iloc[0].directions)
    list_ingredients_trial.append(recipes_df[recipes_df.Key==b].iloc[0].Terms)
    list_user_id_trial.append(user_vectors.keys())

# Create dataframe with ingredients and calories
recipe_recommender_trial = pd.DataFrame({'Recipe_Ingredients':list_ingredients_trial,'Directions':list_directions_trial,'User_id':list_user_id_trial})

# Generate recommendations for all the users 
list_directions = list()
list_ingredients=list()
list_user_id = list()


for i in orders_df_masked['user_id']:
    similar_recipes = u.get_nns_by_vector(user_vectors[i], 5, search_k=-1, include_distances=False)
    for b in similar_recipes:
        list_directions.append(recipes_df[recipes_df.Key==b].iloc[0].directions)
        list_ingredients.append(recipes_df[recipes_df.Key==b].iloc[0].Terms)
        list_user_id.append(i)
        
recipe_recommender_all = pd.DataFrame({'Recipe_Ingredients':list_ingredients,'Directions':list_directions,'User_id':list_user_id})
