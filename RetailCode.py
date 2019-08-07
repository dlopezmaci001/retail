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
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import random
import re
import json

# CREATE RECIPES DATASET

# import recipes. (https://www.kaggle.com/hugodarwood/epirecipes)
with open('C:/Users/Daniel/Desktop/Master/Term 3/Retail/full_format_recipes.json') as json_data:
    new_recipes = json.load(json_data)
    json_data.close()  

# Create a Dataframe with the recipes 
keys = list(new_recipes[0].keys()) # get important keys
df = pd.DataFrame.from_dict(new_recipes, orient='columns')
df['Key'] = range(1, len(df)+1,1)
df['Terms'] = df['ingredients']

# create calories and recipes lists
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
products = pd.read_csv("C:/Users/Daniel/Desktop/Master/Term 3/Retail/products.csv")

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
This generates a vector for each word present in the table
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
         

# Generate product vectors.
# Rename the columns of the product table 
product_table = product_table.rename(index=str, columns={"Products": "Key", "prod_words": "Terms"})

# Generate a dictionary with the average vector of all the products
product_vectors = doc_vectors(word_vectors, product_table)  
"""
This generates a dictionary with all the average vectors for all the products 
there are available, in both of the recipes and products sold by the vendor
"""

# =============================================================================
#                                      No sé por qué se hace esto 
# =============================================================================

# create recipes dataframe
recipes_df = pd.DataFrame({'Key':range(1, len(recipe_list)+1,1), 'ingredients':recipe_list,
                                         'directions':directions_list}) 
    
recipes_df = recipes_df.rename(index=str, columns={"ingredients": "Terms"})

# Remove empty terms

recipes_df = recipes_df[recipes_df.astype(str)['Terms'] != '[]']

# Remove all rows which list contain '' term
#recipes_df.drop(recipes_df[recipes_df['Key'] == 3680].index, inplace=True)
#
#recipes_df.drop(recipes_df[recipes_df['Key'] == 5210].index, inplace=True)
#
#recipes_df.drop(recipes_df[recipes_df['Key'] == 10797].index, inplace=True)


# To detect where code breaks
# for doc in recipes_df.iterrows():
#        
#        doc_vector = list()
#        a = list()
#        for term in doc[1]['Terms']:
#            
#            a.append(term)
#            doc_vector.append(product_vectors[term])
#            print(recipes_df['Key'],term)

## Separate all words with a ' ' space
#recipes_df['Terms']  = recipes_df['ingredients'].str.replace('\W', ' ')
#
## Split products into terms: Tokenize.
#recipes_df['Terms'] = recipes_df['Terms'].str.split()
#
recipe_vectors = doc_vectors(product_vectors, recipes_df)
#
# =============================================================================

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
        

