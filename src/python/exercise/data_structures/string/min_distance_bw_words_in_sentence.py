# def min_distance_bw_words_in_sentence (word_list : list,word1 : str,word2:str) -> int:

def find_min_distance(word_list:list,word1:str,word2:str):
    min_index = 0
    max_index = 0

    for i,word in enumerate(word_list):
        if word == word1 or word == word2:
            print("Inside the for loop")
            if min_index == 0 :
                min_index = i
                print("The min index is " + str(min_index)  + " and word = " + word)
            else:
                max_index = i
                print("The max index is " + str(max_index) + " and word = " + word)
                break # Exit the loop after finding the second word
    return max_index - min_index

words_list = ["hen", "the", "fox", "cf", "cf", "vdf", "and", "bed"]
word1 = "the"
word2 = "and"

print(find_min_distance(words_list,word1,word2))









