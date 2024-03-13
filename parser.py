from inout import InOut

EXCEPTIONS = ['mr.', 'mrs.', 'dr.', 'jr.', 'sr.', 'prof.', 'st.', 'ave.', 'blvd.', 'rev.']
io_tool = InOut()


class Parser:
    def __init__(self):
        self.text = None
        self.words = None
        self.sentences_dict = dict()
        self.cleaned_words = list()
        self.sentences_list = list()

    def take_input_and_split(self, inp):
        # reset words list
        self.words = list()
        # store input text and split it into words
        self.text = inp
        self.words = self.text.split()

    def input_to_sentences(self):
        # reset sentences dict and list
        self.sentences_dict = dict()
        self.sentences_list = list()
        # create a sentence number for the dictionary key and an empty sentence string
        sentence_num = 1
        sentence = ""
        # loop through each word in the input
        for word in self.words:
            # add the word to the sentence placeholder
            sentence = sentence + f"{word} "
            # loop through each character in each word
            for c in word:
                # if the character is a sentence-ending punctuation mark, add the sentence to the dictionary
                # "sentences"
                if c in ['.', '!', '?']:
                    # IF word is in exceptions, move on to the next word. ELSE, end the sentence.
                    if word.lower() in EXCEPTIONS:
                        pass
                    else:
                        sentence = sentence.strip()
                        # store sentence and punctuation in sentences dictionary
                        self.sentences_dict.update({sentence_num: {
                            # removes the punctuation from the sentence
                            "sentence_string": sentence.rstrip(sentence[-1]),
                            "ending_punctuation": c
                        }})
                        # store sentence with punctuation in sentences list
                        self.sentences_list.append(str(sentence))
                        # reset the sentence placeholder and increase the sentence number by 1
                        sentence = ""
                        sentence_num += 1
                # if not, move on to the next word
                else:
                    pass

    def parser_main_loop(self):
        # main parser loop. allows user to enter text through typing or a file input
        while True:
            # user choose mode or exit with a space. handles any cases type (type vs TYPE both work)
            mode = input("To type, enter 'type'\n"
                         "To use a file, enter 'file'\n"
                         "To exit, enter a space (' ')\n")
            if mode.lower() == "type":
                # if user chooses type, let them type and split it.
                self.take_input_and_split(input("Enter your text:\n"))
            elif mode.lower() == "file":
                # if user chooses file, try to receive a file path. handle exception where file path does not exist.
                try:
                    # take file path, read file contents, split file contents
                    io_tool.take_file_path(path=input("Enter your input file path: "), file_type=0)
                    io_tool.read_input_file()
                    self.take_input_and_split(io_tool.input_file_text)
                except FileNotFoundError:
                    # file not found, tell user that their path did not work and start the loop again.
                    print("File not found. Please try again.")
                    continue
            elif mode.lower() == " ":
                # if user chooses space, end process.
                print("Space selected. Parser exited.")
                break
            else:
                # any other user entry is not valid, so prompt them to choose one of the options and start the loop.
                print("Please enter 'type, 'file', or a space (' ').")
                continue
            if len(self.words) < 1:
                # if user entered less than 1 word, ask them to enter text.
                print("Please input some text.")
                continue
            else:
                # if user entered more than one word:
                # clean text (see method)
                self.clean_text(self.words)
                print(self.cleaned_words)
                # take input and split into sentences in the sentence dict and list
                self.input_to_sentences()
                if len(self.sentences_dict) < 1:
                    # if no sentences were entered (no punctuation used), tell user
                    print("No sentences inputted. Make sure to use punctuation!")
                else:
                    # if 1+ sentences entered, print sentence dict and list
                    print(self.sentences_dict)
                    print(self.sentences_list)
                # ask user if they would like to record to output file
                self.opt_for_output()
            # after output, determine if user would like to move on to the Lexer, or parse new data
            while True:
                opt_for_lex = input("Move on to the Lexer?\n"
                                    "'Y' for Yes\n"
                                    "'N' for No\n")
                if opt_for_lex.upper() == "Y":
                    # if yes, break this loop and move to break outside of this loop to break the parser main loop
                    print("Yes selected. Moving on to the Lexer.")
                    break
                elif opt_for_lex.upper() == "N":
                    # if no, call parser main loop
                    print("No selected. Continuing Parser main loop. Parser data will be reset.")
                    self.parser_main_loop()
                    break
                else:
                    # if selection was invalid, continue loop
                    print("User did not input a valid selection. Please try again.")
                    continue
            # break parser main loop
            break

    def clean_text(self, to_clean):
        # reset cleaned words list in case user is doing multiple inputs in a row
        self.cleaned_words = list()
        for word in to_clean:
            # initialize holder for potential edits
            holder = word.lower()
            if holder in EXCEPTIONS:
                # if the word is in exceptions (see EXCEPTIONS), pass. this handles words that
                # have a period in them but aren't at the end of a sentence.
                pass
            else:
                # if word is not an exception
                for c in word:
                    # for character in word
                    if c.isalnum():
                        # if character is alphanumeric, move to next character
                        pass
                    else:
                        # if character is not alphanumeric (likely ending punctuation)
                        if c == "’" or c == "-" or c == "'":
                            # if character is an apostrophe or hyphen, meaning it's not an end of sentence
                            # punctuation mark
                            if c == "’":
                                # format words to all have the same type of apostrophe. handles issues with
                                # file inputs that have different types of apostrophes
                                holder = holder.replace(c, "'")
                            # move on to next word
                            pass
                        else:
                            # if character is not alphanumeric and is an end of sentence punctuation,
                            # replace with blank space so that it can be stripped
                            holder = holder.replace(c, " ")
            # strip word and add to cleaned words list
            self.cleaned_words.append(holder.strip())

    def opt_for_output(self):
        # let user choose if they would like to write to output file, allow user to input output file path
        selection = input("Would you like to write to an output file?\n"
                          "'Y' for Yes\n"
                          "'N' for No\n")
        if selection.upper() == "Y":
            # if yes chosen
            try:
                # try to take file path and write to it
                io_tool.take_file_path(path=input("Enter your output file path: "), file_type=1)
                io_tool.write_lines_to_output_file(self.cleaned_words)
                io_tool.write_lines_to_output_file(self.sentences_list)
                print("Successfully wrote to output file.")
            except FileNotFoundError:
                # if file path not found
                print("File path error. Please try again.")
                # call opt_for_output to let user try again.
                self.opt_for_output()
        elif selection.upper() == "N":
            # if no chosen
            print("User chose to not write to an output file.")
        else:
            # if no option selected
            print("No option selected. Please try again.")
            # call opt_for_output to let user try again.
            self.opt_for_output()
