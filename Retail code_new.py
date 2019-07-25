# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:55:45 2019

ANALYTICS FOR RETAIL AND CONSUMER GOODS
FINAL PROJECT
TEAM: Beatriz Asenjo, Tamara Samman, Daniel Lopez, Rafael Bastardo
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
with open('C:/Users/Daniel/Desktop/Retail/full_format_recipes.json') as json_data:
    new_recipes = json.load(json_data)
    json_data.close()

# fill recipes dataframe
keys = list(new_recipes[0].keys()) # get important keys
col_names = list(['recipe_id','ingredients','calories'])

# create calories and recipes lists
recipe_list = list()
calories_list = list()
for i in range(0,len(new_recipes)):
    try:
        recipe = list(new_recipes[i]['ingredients'])
        words_to_eliminate = ['cups','cup','spoon','spoons','teaspoon','teaspoons','tablespoon','tablespoons','pound','pounds','lb']
        remove = '|'.join(words_to_eliminate)
        recipe = [re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE).sub("",w) for w in recipe]
        recipe = [re.sub(r'\([^)]*\)', "", w) for w in recipe]
        recipe = [re.sub(r'[\[\],\')Â¿?^*"(\.;$%_]', "", w) for w in recipe]
        recipe = [re.sub(r'[0-9/,-]', "", w) for w in recipe]
        recipe = [re.sub(r'^\s+', "", w) for w in recipe]
        recipe = [re.sub(r'\s+', " ", w) for w in recipe]
        try:
            calories_list.append(float(new_recipes[i]['calories']))
            recipe_list.append(recipe)
        except:
            print('no calories in recipe')
    except:
        print('no ingredients in recipe')
        
# create recipes dataframe
recipes = pd.DataFrame({'recipe_id':range(1, len(recipe_list)+1,1),'calories':calories_list,
                        'ingredients':recipe_list})

# CREATE PRODUCTS DATAFRAME
products = pd.read_csv("C:/Users/Daniel/Desktop/Retail/products.csv")

# Put into a list 
product_list = list(products['product_name'])
products.head()

### Recipe upluad 
#with open('C:/Users/Daniel/Desktop/Retail/train.json') as json_data:
#    d = json.load(json_data)
#    json_data.close()
#
### CREATE RECIPES DATASET
#recipes_cuisine     = list()
#recipes_id          = list()
#recipes_ingredients = list()
#
#for element in d:
#        recipes_cuisine.append(element['cuisine'])
#        recipes_id.append(element['id'])
#        recipes_ingredients.append(element['ingredients'])
#
#recipes = pd.DataFrame()
#recipes['cuisine']=recipes_cuisine
#recipes['id']=recipes_id
#recipes['ingredients']=recipes_ingredients
#recipes.head()

# OBTAIN INGREDIENTS IN RECIPE
list_prods = list()
for recipe in recipes['ingredients']:
    list_prods += recipe
    
# KEEP ONE EXAMPLE OF EACH    
products_recipes = list(set(list_prods)) # set deletes duplicates
display(len(products_recipes))
display(len(list_prods))
display(list_prods[:10])

# TRAIN WORD2VEC

# Create list of terms that is going to be used to train the word2vec model. 
combined_list = product_list + products_recipes

display('Number of products from dataset: {}'.format(len(list(products.product_name))))
display('Number of ingredients from dataset: {}'.format(len(products_recipes)))
display('Total number of ingredients + products from the recipes dataset: {}'.format(len(combined_list)))

# Create a table with the prods
product_table = pd.DataFrame(combined_list, columns=['Products'])

product_table['prod_words'] = product_table['Products'].str.lower()

product_table['prod_words']  = product_table['Products'].str.replace('\W', ' ')

# Split products into terms: Tokenize.
product_table['prod_words'] = product_table['Products'].str.split()


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


#Call the function using the list of products.
word_vectors = term_vectors(list(product_table['prod_words']))

# Calculate the average for the vectors

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
product_table = product_table.rename(index=str, columns={"Products": "Key", "prod_words": "Terms"})
product_vectors = doc_vectors(word_vectors, product_table)  

# Generate recipe vectors.
recipes_table = recipes.rename(index=str, columns={"recipe_id": "Key", "ingredients": "Terms"})
recipe_vectors = doc_vectors(product_vectors, recipes_table)


del product_vectors['']    
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

products[products['product_name'].str.contains(" oil ")]

# Create a ficticious cart 

cart = ['apple', 'oil', "flour"]

#cart_vector = doclist_2_vector(product_vectors, cart)

def basket_average(product_vectors, cart):
    basket = list()
    for i in cart:
        basket.append(product_vectors[i])
    vec_avg = np.average(basket, axis=0)
    return vec_avg
    
  
cart_vector = basket_average(product_vectors, cart)

display(cart)
display(cart_vector)

#similar_products = u.get_nns_by_item(10, 10)
similar_recipes = u.get_nns_by_vector(cart_vector, 5, search_k=-1, include_distances=False)
display(similar_recipes)

# Find recipes

#display(similar_baskets)
for b in similar_recipes:
    display(recipes_table[recipes_table.Key==b].calories)
    for ingredient in list(recipes_table[recipes_table.Key==b].Terms):
        #w.similar_by_vector(vector, 5, search_k=-1, include_distances=False)
        print(ingredient)
        
# Create recommender 
        
orders_df = pd.read_csv("C:/Users/Daniel/Desktop/Retail/orders.csv") 

orders_df.head()   

# Visualise the graph
orders_df['user_id'].value_counts().hist(bins=100)

#Calculate the mean
display(np.mean(orders_df['user_id'].value_counts()))
display(np.median(orders_df['user_id'].value_counts()))

# Visualise 
orders_df['days_since_prior_order'].hist(bins=100)

# Average orders
display(np.mean(orders_df['days_since_prior_order']))
# 2 to 3 orders per month

# count group orders
orders_count = orders_df['user_id'].value_counts().to_frame().reset_index().rename(index=str, columns={'user_id':'count'})
display(orders_count[orders_count['count']>10]['index'].count())
#display(orders_count[orders_count['count']>10]['index'].count())

# Best customers

orders_df[orders_df['user_id'].isin(orders_count[(orders_count['count']>30) & (orders_count['count']<32)]['index'])].count()

orders_df_masked = orders_df[orders_df['user_id'].isin(orders_count[(orders_count['count']>10) & (orders_count['count']<15)]['index'])]
display(np.mean(orders_df_masked['days_since_prior_order']))

display('Number of original orders: {:,}'.format(len(orders_df_masked.order_id.unique())))
display('Number of original orders: {:,}'.format(len(orders_df.order_id.unique())))

# Masked orders combine


orders_df_masked = orders_df_masked.groupby('user_id')['order_id'].apply(list)
orders_df_masked = orders_df_masked.reset_index()#.head()
orders_df_masked.head()

# concatenate
order_ids = list()
for order in orders_df_masked.order_id:
    order_ids +=order
    
order_products_prior = pd.read_csv("C:/Users/Daniel/Desktop/Retail/order_products__prior.csv")

# Baskets 

baskets = order_products_prior[order_products_prior['order_id'].isin(order_ids)]
#Group 
baskets_df_masked = baskets.groupby('order_id')['product_id'].apply(list)

# View 
baskets_df_masked.head()

# Create list 
product_id_pair=dict()
for product in products.iterrows():
    product_id_pair[product[1].product_id] = product[1].product_name

# create the basket vector    
import time
basket_vectors = dict()
for basket in baskets_df_masked.iteritems(): 
    vector_list = list()
    
    for product in basket[1]:
        vector_list.append(product_vectors[product_id_pair[product]])
        
    basket_vectors[basket[0]]=np.average(vector_list, axis=0)  
    
    
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
    user_vectors[user[0]]=np.average(vector_list, axis=0)    
    
display(list(user_vectors.items())[:3])    

# Recommender

similar_recipes = u.get_nns_by_vector(user_vectors[3], 50, search_k=-1, include_distances=False)

# looking for list of calories in recipes dataframe

from collections import Counter
list_calories = list()
list_ingredients=list()
for b in similar_recipes:
    list_calories.append(recipes[recipes.recipe_id==b].iloc[0].calories)
    list_ingredients.append(recipes[recipes.recipe_id==b].iloc[0].ingredients)

# Create dataframe with ingredients and calories
recipe_recommender = pd.DataFrame({'Recepe_Ingredients':list_ingredients,'Calories':list_calories})

# Order data using the min and max number of calories specified by the customer
min_max=[250,500]
recipe_recommender[recipe_recommender.Calories>min_max[0]][recipe_recommender.Calories<=min_max[1]].sort_values('Calories',ascending=False)

