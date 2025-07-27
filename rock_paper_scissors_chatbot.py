# import LM_Function
import random
import re
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

intents = {
            "Rock": ["Rock", "I pick Rock", "I choose Rock", "I will go with Rock"],
            "Paper": ["Paper", "I pick Paper", "I choose Paper", "I will go with Paper"],
            "Scissors": ["Scissors", "I pick Scissors", "I choose Scissors", "I will go with Scissors"],
            "Rules": ["Rules", "Rules?", "What Rules?", "What are the Rules?"],
        }

responses = {
            "Rock":"Ah Rock!",
            "Paper":"Ah Paper",
            "Scissors":"Ah Scissors",
            "Rules": "user inputs one of three gestures: rock, paper , or scissors. Rock crushes scissors, scissors cuts paper, and paper covers rock. AI will simultaneously pick an option as well. The player with the winning gesture wins the round. "
            }      

# 0 = Rock, 1 = Paper, 2 = Scissors
aiOutputs = [0, 1, 2] 
aiOutputsText = ["Rock", "Paper", "Scissors"] 

userScore = 0
aiScore = 0

vectorizer = TfidfVectorizer()

# Sample function for intent matching
def match_intent(user_input):
    # Flatten the intents dictionary to create a list of all possible phrases
    intent_phrases = [phrase for phrases in intents.values() for phrase in phrases]
    
    # Fit the vectorizer on the phrases and transform user input
    all_phrases = intent_phrases + [user_input] # Append user input to the list
    vectors = vectorizer.fit_transform(all_phrases)
    
    # Compute cosine similarities between user input and each intent phrase
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()  # Last vector is user input
    # Find the index of the highest similarity score
    # print(f" --- {cosine_similarities[cosine_similarities.argmax()]} --- ")
    if cosine_similarities[cosine_similarities.argmax()] > 0.3:
        best_match_index = cosine_similarities.argmax()
        best_match_phrase = intent_phrases[best_match_index]
        
        # Map the best matching phrase back to its intent
        for intent, phrases in intents.items():
            if best_match_phrase in phrases:
                #print(f" --- {intent} --- ")
                return intent
    else:    
        return 'error'

def match_winner(user_input, ai_input):
    # Return updated scores and winner

    global aiScore, userScore

    if user_input == ai_input:
        return "Draw."
    
    if (user_input == 0) and (ai_input == 1):
        aiScore+=1
        return "I win"
    if (user_input == 0) and (ai_input == 2):
        userScore+=1
        return "You win."
    
    if (user_input == 1) and (ai_input == 0):
        userScore+=1
        return "You win."
    if (user_input == 1) and (ai_input == 2):
        aiScore+=1
        return "I win."
    
    if (user_input == 2) and (ai_input == 0):
        userScore+=1
        return "You win."
    if (user_input == 2) and (ai_input == 1):
        aiScore+=1
        return "I win."
    
def get_response(user_input):

    intent = match_intent(user_input)
    aiTurn = random.randint(0,2)

    if intent == 'Rock':
        print("Bot:", responses['Rock'])
        result = match_winner(0, aiTurn)
        print("Bot: I choose " + aiOutputsText[aiTurn] + ". " + str(result))
        print("Current score: User:" + str(userScore) + " | AI:" + str(aiScore))
    elif intent == 'Paper':
        print("Bot:", responses['Paper'])
        result = match_winner(1, aiTurn)
        print("Bot: I choose " + aiOutputsText[aiTurn] + ". " + str(result))
        print("Current score: User:" + str(userScore) + " | AI:" + str(aiScore))
    elif intent == 'Scissors':
        print("Bot:", responses['Scissors'])
        result = match_winner(2, aiTurn)
        print("Bot: I choose " + aiOutputsText[aiTurn] + ". " + str(result))
        print("Current score: User:" + str(userScore) + " | AI:" + str(aiScore))
    elif intent == 'Rules':
        print("Bot:", responses['Rules'])
    else:
        print("Bot:", "Sorry I didn't get that.") 

# Main function to run chatbot
def main() :
    print("Bot: Welcome to the Rock, Paper, Scissors Chatbot. Enter your choice and lets play!")
    while True :
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print ("Bot:","Goodbye!")
            break
        get_response(user_input)

if __name__ == "__main__":
    main()