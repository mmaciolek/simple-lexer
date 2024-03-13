import parser
import random
import nltk
from nltk.corpus import brown
import inout
import math
from nltk import ConditionalFreqDist, ngrams
from gensim.models import Word2Vec
import logging
import string


class Lexer(parser.Parser):
    def __init__(self):
        super().__init__()
        self.lang_dict = dict()

    def main_loop(self):
        # call parser main loop to allow user to parse data
        self.parser_main_loop()
        # lexer loop starts here
        while True:
            # if number of words is 0, break loop
            if len(self.cleaned_words) < 1:
                break
            # allow user to choose an analysis method
            selection = input("Choose an analysis method by entering the corresponding number:\n"
                              "'1' - Maximum Likelihood Estimation\n"
                              "'2' - Brown Corpus Parts of Speech\n"
                              "'3' - Early Machine Translation\n"
                              "'4' - Information Retrieval\n"
                              "'5' - N-gram Model\n"
                              "'6' - Word Embeddings\n"
                              "'7' - Bayesian Methods\n"
                              "' ' - Enter a space to exit\n"
                              "Enter your selection: ")
            if selection == "1":
                self.maximum_likelihood_estimation()
            elif selection == "2":
                self.brown_corpus_parts_of_speech()
            elif selection == "3":
                self.early_machine_translation()
            elif selection == "4":
                self.information_retrieval()
            elif selection == "5":
                self.n_gram_model()
            elif selection == "6":
                self.word_embeddings()
            elif selection == "7":
                self.bayesian_methods()
            elif selection == " ":
                print("Space selected. No analysis chosen.")
                # if user chooses space, ask if they want to parse new data
                while True:
                    new_parse = input("Parse new data?\n"
                                      "'Y' for Yes\n"
                                      "'N' for No\n")
                    if new_parse.upper() == "Y":
                        # if user wants to parse new data, call lexer main loop
                        # this prompts a call of the parser main loop
                        self.main_loop()
                    elif new_parse.upper() == "N":
                        # if no, terminate process by breaking the loop
                        print("No selected. Process terminated.")
                        break
                    else:
                        # if input is invalid, ask user to try again and continue loop
                        print("User inputted invalid selection. Please try again.")
                        continue
                # break lexer main loop
                break
            # if user analysis method selection was invalid, ask user to select a number 1-6
            else:
                print("Please select a number 1-7.")

    def maximum_likelihood_estimation(self):
        # create list of bigrams by grouping each word with the word that follows it
        bigrams_list = list()
        for _ in range(len(self.cleaned_words) - 1):
            bigrams_list.append((self.cleaned_words[_], self.cleaned_words[_ + 1]))
        # count each unique word
        word_count = dict()
        for word in set(self.cleaned_words):
            word_count[word] = self.cleaned_words.count(word)
        # count each unique bigram
        bigram_count = dict()
        for bigram in set(bigrams_list):
            bigram_count[bigram] = bigrams_list.count(bigram)
        # determine probability of each bigram by dividing the bigram frequency by the count for the first word in
        # the bigram
        bigram_probabilities = dict()
        for bigram, count in bigram_count.items():
            bigram_probabilities[bigram] = count / word_count[bigram[0]]
        # try to get a sentence length from the user between 5 and 15
        try:
            sentence_length = int(input("Enter a sentence length between 5 and 15.\n"))
            if 5 <= sentence_length <= 15:
                # if sentence length between 5 and 15
                def generate_sentence(current_word, length=sentence_length):
                    # initialize the sentence list with the first word
                    sentence = [current_word.title()]
                    # loop with range equal to desired sentence length - 1
                    for _ in range(length - 1):
                        # determine next words by finding all bigrams with the current word as the first word
                        # collect all the following words into 'next_words'
                        next_words = list()
                        for bigram_prob in bigram_probabilities:
                            if bigram_prob[0] == current_word:
                                next_words.append(bigram_prob[1])
                        # for all the next words, determine the probability of its pair with the current word
                        next_word_probabilities = list()
                        for next_word in next_words:
                            next_word_probabilities.append(bigram_probabilities[(current_word, next_word)])
                        # if no bigrams are available, break loop
                        if not next_words:
                            print("No bigram available. Unable to generate up to the desired sentence length.")
                            break
                        # pick the next word by using the next word probabilities to select one of the next words
                        next_word = random.choices(next_words, weights=next_word_probabilities)[0]
                        # if last iteration of the loop
                        if _ == length - 2:
                            # add a period to the word
                            next_word = next_word + "."
                        sentence.append(next_word)
                        current_word = next_word
                    # join the words together with spaces between them and return the string
                    return ' '.join(sentence)
                # choose a random start word
                # only pick from a list that includes all words except the final word, since it will
                # not have a bigram with it as the first word.
                start_word = random.choice([word for word in self.cleaned_words[:-1]])
                generated_sentence = generate_sentence(start_word)
                print("Generated sentence: ", generated_sentence)
            else:
                # if invalid sentence length, print error and call method to try again
                print("Invalid sentence length. Please try again.")
                self.maximum_likelihood_estimation()
        except TypeError:
            # if user inputted incorrect data type, print error message and call method to try again
            print("Invalid sentence length data type. Please try again.")
            self.maximum_likelihood_estimation()
        except ValueError:
            # if user inputted an invalid sentence length value, print error message and call method to try again
            print("Invalid sentence length value entered. Please try again.")
            self.maximum_likelihood_estimation()

    def brown_corpus_parts_of_speech(self):
        # if not downloaded, download universal tag set
        nltk.download('universal_tagset')
        # create dictionary of all words with their respective tags
        brown_words_tags_dict = dict((word.lower(), tag) for word, tag in brown.tagged_words(tagset='universal'))
        # initialize sets of each word tag
        adjectives = set()
        adpositions = set()
        adverbs = set()
        conjunctions = set()
        determiners_articles = set()
        nouns = set()
        numerals = set()
        particles = set()
        pronouns = set()
        verbs = set()
        other = set()
        unidentified = set()
        # iterate through each word, determining its tag and placing it in the respective set
        for word in self.cleaned_words:
            if word in brown_words_tags_dict.keys():
                part_of_speech = brown_words_tags_dict[word]
                if part_of_speech == "ADJ":
                    adjectives.add(word)
                elif part_of_speech == "ADP":
                    adpositions.add(word)
                elif part_of_speech == "ADV":
                    adverbs.add(word)
                elif part_of_speech == "CONJ":
                    conjunctions.add(word)
                elif part_of_speech == "DET":
                    determiners_articles.add(word)
                elif part_of_speech == "NOUN":
                    nouns.add(word)
                elif part_of_speech == "NUM":
                    numerals.add(word)
                elif part_of_speech == "PRT":
                    particles.add(word)
                elif part_of_speech == "PRON":
                    pronouns.add(word)
                elif part_of_speech == "VERB":
                    verbs.add(word)
                elif part_of_speech == ".":
                    pass
                elif part_of_speech == "X":
                    other.add(word)
                else:
                    print("Part of speech error.")
            else:
                unidentified.add(word)
        keys = ['adjectives', 'adpositions', 'adverbs', 'conjunctions', 'determiners_articles', 'nouns', 'numerals',
                'particles', 'pronouns', 'verbs', 'other', 'unidentified']
        value_lists = [adjectives, adpositions, adverbs, conjunctions, determiners_articles, nouns,
                       numerals, particles, pronouns, verbs, other, unidentified]
        # create a dictionary of the tag types and the words in their set
        output = dict(zip(keys, value_lists))
        # display the values in each tag set, making sure to let the user know if there are none in the set
        for part in output.items():
            if len(part[1]) < 1:
                print(f"No words of type '{part[0]}' found.")
            else:
                print(part)

    def early_machine_translation(self):
        print("This analysis section will attempt to translate and display as many words from your input "
              "as possible.")
        while True:
            langs = ['spanish', 'french', 'italian', 'german']
            # allow user to choose a language
            selection = input("Please choose a language:\n"
                              "'Spanish'\n"
                              "'French'\n"
                              "'Italian'\n"
                              "'German'\n"
                              "' ' Enter a space to exit.\n")
            if selection.lower() in langs:
                # if language is selected
                in_out = inout.InOut()
                # read language file into lang_dict
                self.lang_dict = in_out.read_language(f'files/langs/{selection.lower()}.txt')
                translated_words = set()
                # loop through cleaned words
                for word in self.cleaned_words:
                    if word in self.lang_dict.keys():
                        # if word is in the language dictionary, add it to the translated words set
                        translated_words.add(word)
                print("Words translated:\n")
                for word in translated_words:
                    # for word in the translated words set, print the word and its translation
                    print(f"{word.title()}: {self.lang_dict[word.lower()]}")
            elif selection == ' ':
                # if space selected
                print("Space selected. No translation.")
                break
            else:
                # if invalid input, prompt user to choose one of the languages
                print("Please select one of the available languages.")

    def information_retrieval(self):
        # retrieve sentences without punctuation to use as documents
        documents = [sentence_info['sentence_string'] for (num, sentence_info) in self.sentences_dict.items()]
        # ask for query
        user_query = input("Enter a query:\n")
        # if user does not enter a query, prompt user to enter a valid query and call method to try again
        if len(user_query) < 1:
            print("Please enter a valid query.")
            self.information_retrieval()
        else:
            # if query is valid

            def compute_tf(test_word, doc):
                # compute term frequency
                words = doc.split()
                word_count = len(words)
                frequency = 0
                # count how many times the test word is in the doc
                for word in words:
                    if word == test_word:
                        frequency += 1
                # return the frequency divided by the word count
                return frequency / word_count

            def compute_idf(test_word, docs):
                # compute inverse document frequency
                num_docs = len(docs)
                docs_with_word = 0
                # loop through each doc
                for doc in docs:
                    # if test word is in the doc, add 1 to docs_with_word
                    if test_word in doc:
                        docs_with_word += 1
                # return log of number of docs divided by total docs with the word
                return math.log(num_docs / docs_with_word)

            def compute_tf_idf(doc, docs):
                # compute if-idf
                tf_idf = dict()
                # loop through each unique word in the document
                for word in set(doc.split()):
                    # get term frequency
                    tf = compute_tf(word, doc)
                    # get inverse document frequency
                    idf = compute_idf(word, docs)
                    # calculate tf-idf
                    tf_idf[word] = tf * idf
                return tf_idf

            def score_query(query, doc, docs):
                # calculate score
                score = 0.0
                # loop through each word in the query
                for word in query.split():
                    # if word is in the doc
                    if word in doc:
                        # get the tf-idf score for the word
                        score += compute_tf_idf(doc, docs).get(word, 0)
                return score
            # create dictionary of scores and print
            scores = {doc: score_query(user_query, doc, documents) for doc in documents}

            print(scores)

    def n_gram_model(self):
        while True:
            # limit word list to only alphabetic words
            alphabetic_words = [word.lower() for word in self.cleaned_words if word.isalpha()]

            n = int()
            # initialize cfd variable to hold frequency for either mode
            cfd = None
            user_word = None
            # if total words is less than 3, automatically use bigrams
            if len(alphabetic_words) < 3:
                n = 2
            else:
                # if total words is greater than 2, allow user to choose a mode
                selection = input("Choose a mode.\n"
                                  "'b' for Bigrams\n"
                                  "'t' for Trigrams\n"
                                  "Enter a space (' ') to exit.\n")
                if selection.lower() == 'b':
                    # if bigrams chosen, take user input
                    user_word = input("Please input a word.\n"
                                      "Enter a space (' ') to exit.\n")
                    if user_word == " ":
                        # if user enters space, exit n gram analysis
                        print("Space selected. No N-gram analysis will be done.")
                        break
                    elif len(user_word.split()) > 1:
                        # if user enters multiple words, recall the method and try again
                        print("Error. Too many words entered. Please try again.")
                        self.n_gram_model()
                        break
                    else:
                        # else, create n gram list and generate conditional frequency distribution
                        n = 2
                        ngrams_list = list(ngrams(alphabetic_words, n))
                        cfd = ConditionalFreqDist(ngrams_list)
                elif selection.lower() == 't':
                    # if trigrams selected, allow user to enter input
                    user_word = input("Please input a two word string.\n"
                                      "Enter a space (' ') to exit.\n")
                    if user_word == " ":
                        # if user chooses space, end n gram analysis
                        print("Space selected. No N-gram analysis will be done.")
                        break
                    elif len(user_word.split()) != 2:
                        # if user enters a number of words that isn't 2, call method to try again
                        print("Error. User entered an invalid number of words. Please try again.")
                        self.n_gram_model()
                        break
                    else:
                        # generate trigram list and calculate conditional frequency distribution
                        n = 3
                        ngrams_list = list(ngrams(alphabetic_words, n))
                        cfd = ConditionalFreqDist(((word1, word2), word3) for word1, word2, word3, in ngrams_list)
                elif selection == " ":
                    # space selected, no n gram analysis. break loop
                    print("Space selected. No N-gram analysis will be done.")
                    break
                else:
                    # mode selection input invalid. call method to try again
                    print("Error during mode selection. Please try again.")
                    self.n_gram_model()
                    break

            def predict_next_word(word):
                # if n is 2, format word to catch any errors
                if n == 2:
                    word = word.lower().split()[0]
                else:
                    # if n is 3, format words into a tuple that can be used to find the cfd
                    word = tuple(word.lower().split()[0:2])
                if word in cfd:
                    # find the most common next word and return it
                    most_common_next = cfd[word].max()
                    return most_common_next
                else:
                    return "No prediction available"

            if len(user_word) < 1:
                print("No input detected. Please try again.")
                continue
            predicted_word = predict_next_word(user_word)
            print(f"Your word was: {user_word}\n"
                  f"The predicted next word was: {predicted_word}")

    def word_embeddings(self):
        # enable logging to see potential issues with model creation
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # create list of sentences without punctuation
        sentences = [sen_data['sentence_string'].split() for (num, sen_data) in self.sentences_dict.items()]
        # create model
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        while True:
            # ask user to input a string to create a vector
            user_input = input("Input a string to create the vector with.\n"
                               "Enter a space (' ') to exit.\n")
            if user_input == " ":
                # if user inputs a space, break the loop
                print("Space selected. No vector will be created.")
                break
            else:
                # if user inputs a string, try to create a vector
                try:
                    # create vector and print it
                    vector = model.wv[user_input]
                    print(f"Vector for {user_input}:\n",
                          vector)
                    # find 3 most similar words and print them
                    similar_words = model.wv.most_similar(user_input)[:3]
                    print(f"Words similar to {user_input}\n", similar_words)
                    # find 3 most dissimilar words and print them
                    dissimilar_words = model.wv.most_similar(user_input)[-3:]
                    print(f"Words dissimilar to {user_input}\n", dissimilar_words[::-1])
                except KeyError:
                    # key error in vector, print error message
                    print("Error. Vector unable to be created from the input.")

    def bayesian_methods(self):
        while True:
            # sample data, list of tuples containing sentences and their ending punctuation
            data = [(info['sentence_string'], info['ending_punctuation']) for num, info in self.sentences_dict.items()]
            # split data into words and keep track of ending punctuation in the sentence associated with the word
            word_counts = {
                ".": {},
                "!": {},
                "?": {}
            }
            total_words = {
                ".": 0,
                "!": 0,
                "?": 0
            }
            for sentence, punctuation_mark in data:
                for word in sentence.lower().split():
                    if word not in word_counts[punctuation_mark]:
                        word_counts[punctuation_mark][word] = 0
                    word_counts[punctuation_mark][word] += 1
                    total_words[punctuation_mark] += 1

            # calculate probabilities for a word being in a sentence with each punctuation mark
            word_probs = {
                ".": {},
                "!": {},
                "?": {}
            }
            for punctuation_mark in word_counts:
                for word in word_counts[punctuation_mark]:
                    word_probs[punctuation_mark][word] = (word_counts[punctuation_mark][word] /
                                                          total_words[punctuation_mark])

            def identify(user_sentence):
                period_prob = exclamation_prob = question_prob = 1.0
                # determine scores
                for user_word in user_sentence.lower().split():
                    if user_word in word_probs["."]:
                        period_prob *= word_probs["."][user_word]
                    if user_word in word_probs["!"]:
                        exclamation_prob *= word_probs["!"][user_word]
                    if user_word in word_probs["?"]:
                        question_prob *= word_probs["?"][user_word]
                # return score
                if period_prob < exclamation_prob and period_prob < question_prob:
                    return "."
                elif exclamation_prob < period_prob and exclamation_prob < question_prob:
                    return "!"
                elif question_prob < period_prob and question_prob < period_prob:
                    return "?"
                else:
                    return "No prediction."
            user_input = input("Please type a sentence that doesn't have an ending punctuation mark.\n")
            if user_input[-1] in set(string.punctuation):
                user_input = user_input[:-1]
            print(f"{identify(user_input)} is the predicted punctuation for your sentence.")
