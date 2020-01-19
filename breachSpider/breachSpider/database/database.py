import pymongo
import threading
from bson.objectid import ObjectId

# This sets the connection to the mongodb server
conn_str = 'mongodb://localhost:27017'
client = pymongo.MongoClient(conn_str)

# Connects to a database, everything after the dot is the database name.
# Changing the database name will create a new database.  Useful for testing
db = client.breachDB


# -----------------Adders--------------

# Adds to the database in the same thread.
# BaseURL - a String containing the base url of a page
# FullURL - a String containing the full url of a page
# ScrapyRedirectURL - a String containing the url that scrapy redirected to if it did, might be the same as FullURL
# FullText - a String of all the text on the page
# LemmatizedText - a String of all the lemmatized text on the page
# Breach - a Boolean indicating if the ML algorithm thinks this is a breach report or not.
# True for breach, False for not breach
# Confirmed - a Boolean indicating if an analyst has confirmed this report or not.
# True for confirmed by an analyst, false for not
# result - Returns the database add result
def db_add(BaseURL, FullURL, ScrapyRedirectURL, FullText, LemmatizedText, Breach, conf, Confirmed = False):
    try:
        # We need to check for previously scraped results of the same page
        fullURL = db_find_by_full_url(FullURL)
        scrapyURL = db_find_by_scrapy_redirect_url(ScrapyRedirectURL)

        # If the page has already been scraped AND confirmed, then we want to keep it and not update it
        if fullURL is not None and fullURL['Confirmed'] or scrapyURL is not None and scrapyURL['Confirmed']:
            return True

        # Need to remove any previously scraped entries
        # We are assuming a new scrape of the same page means the data has been updated
        db.breaches.delete_many({'FullURL': FullURL})
        db.breaches.delete_many({'ScrapyRedirectURL': ScrapyRedirectURL})

        # DB wants Breach to be in Boolean format, but sometimes it is easier to pass a 0 or 1 to the add instead
        actualBreach = False
        if Breach == 0:
            actualBreach = False
        elif Breach == 1 and conf >= .6:
            actualBreach = True
        elif Breach == 1 and conf <.6:
            actualBreach = False
        else:
            actualBreach = Breach

        # DB wants Confirmed to be in Boolean format, but sometimes it is easier to pass a 0 or 1 to the add instead
        actualConfirmed = False
        if Confirmed == 0:
            actualConfirmed = False
        elif Confirmed == 1:
            actualConfirmed = True
        else:
            actualConfirmed = Confirmed

        # Add to the DB
        result = db.breaches.insert_one(
        {
            "BaseURL": BaseURL,
            "FullURL": FullURL,
            "ScrapyRedirectURL": ScrapyRedirectURL,
            "FullText": FullText,
            "LemmatizedText": LemmatizedText,
            "ConfidenceValue": conf,
            "Breach": actualBreach,
            "Confirmed": actualConfirmed
        })
        return result
    except Exception as e:
        print(e)
        return None


# Runs a thread to handle the add, otherwise same as db_add but assumes Confirmed is False
def threaded_db_add(BaseURL, FullURL, ScrapyRedirectURL, FullText, LemmatizedText, Breach, conf):
    t = threading.Thread(target=db_add, args=(BaseURL, FullURL, ScrapyRedirectURL, FullText, LemmatizedText, Breach, conf))
    t.start()


# -----------------Single Getters--------------

# Find a db object with the specified db id
# id - a String containing the _id of the mongo object to search on
# return - Returns a Python dictionary containing the mongo document matching the query
def db_find_by_id(id):
    try:
        result = db.breaches.find_one({"_id": ObjectId(id)})
        return result
    except Exception as e:
        print(e)
        return None


# Find a db object with the specified FullURL
# FullURL - a String containing the FullURL of the mongo object to search on
# return - Returns a Python dictionary containing the mongo document matching the query
def db_find_by_full_url(FullURL):
    try:
        result = db.breaches.find_one({"FullURL": FullURL})
        return result
    except Exception as e:
        print(e)
        return None


# Find a db object with the specified scrapy redirect url
# ScrapyRedirectURL - a String containing the ScrapyRedirectURL of the mongo object to search on
# return - Returns a Python dictionary containing the mongo document matching the query
def db_find_by_scrapy_redirect_url(ScrapyRedirectURL):
    try:
        result = db.breaches.find_one({"ScrapyRedirectURL": ScrapyRedirectURL})
        return result
    except Exception as e:
        print(e)
        return None

# -----------------Multiple Getters--------------
# Some useful references
#    See "Querying for More Than One Document
#    https://api.mongodb.com/python/current/tutorial.html
#
#    https://api.mongodb.com/python/current/api/pymongo/cursor.html# pymongo.cursor.Cursor


# Finds all the db objects matching the entered breach status
# Breach - a Boolean indicating what Breach status to search on
# return -  Returns a list of python dictionaries containing the mongo documents matching the query
def db_find_by_breach_status(Breach):
    try:
        results = db.breaches.find({"Breach": Breach})
        return results
    except Exception as e:
        print(e)
        return None


# Finds all the db objects matching the entered confirmed status
# Confirmed - a Boolean indicating what Confirmed status to search on
# return -  Returns a list of python dictionaries containing the mongo documents matching the query
def db_find_by_confirmed_status(Confirmed):
    try:
        results = db.breaches.find({"Confirmed": Confirmed})
        return results
    except Exception as e:
        print(e)
        return None

# Finds all the db objects matching the entered confirmed status
# Confirmed - a Boolean indicating what Confirmed status to search on
# return -  Returns a list of python dictionaries containing the mongo documents matching the query
def db_grab_unconfirmed():
    try:
        results = db.breaches.find({"Breach": True, "Confirmed": False})
        return results
    except Exception as e:
        print(e)
        return None

# Finds all the db objects matching the entered BaseURL
# BaseURL - a String containing the BaseURL to search on
# return -  Returns a list of python dictionaries containing the mongo documents matching the query
def db_find_by_base_url(BaseURL):
    try:
        results = db.breaches.find({"BaseURL": BaseURL})
        return results
    except Exception as e:
        print(e)
        return None


# -----------------Modifiers--------------

# Modify a db object's breach status
# id - a String containing the _id of the mongo object to update
# newStatus - a Boolean indicating what to update to
# return - returns the result from the database
def db_change_breach(id, newStatus):
    try:
        result = db.breaches.update_one({"_id": ObjectId(id)}, {"$set": {"Breach": newStatus}})
        return result
    except Exception as e:
        print(e)
        return None


# Modify a db object's confirmed status
# id - a String containing the _id of the mongo object to update
# newStatus - a Boolean indicating what to update to
# return - returns the result from the database
def db_change_confirmed(id, newStatus):
    try:
        result = db.breaches.update_one({"_id": ObjectId(id)}, {"$set": {"Confirmed": newStatus}})
        return result
    except Exception as e:
        print(e)
        return None
