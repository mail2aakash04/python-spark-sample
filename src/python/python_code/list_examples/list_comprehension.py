import random

def main():

    numbers = [1, 2, 3, 4, 5]

    # simple list comprehension
    squares = [x **2 for x in numbers]
    print(squares)

    # simple list comprehension using condition
    odd_numbers = [x for x in numbers if x%2 !=0]
    print(odd_numbers)

    # filter words which have length less than 4
    word_list = ["apple", "bat", "ball", "cat", "dog", "elephant"]
    short_words = [ word for word in word_list if  len(word) <= 5 ]
    print(short_words)

    #Assignment to multiple variables#
    numbers_list = [1, 2, 3, 4, 5]
    number_pairs = [ (num,num *2) for num in numbers_list]
    print(number_pairs)

    #Transforming elements
    song_names =  ['Neon Lights', 'Pieces', 'Everything']
    lower_case_song_names = [song.lower() for song in song_names]
    print(lower_case_song_names)








if __name__ == "__main__":
    main()