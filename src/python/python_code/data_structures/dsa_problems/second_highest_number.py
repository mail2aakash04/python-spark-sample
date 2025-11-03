def find_second_highest_number(inputList):
    highest_element = 0
    second_highest_element = 0

    for i in range (len(inputList)-1):
        if inputList[i] > inputList[i+1]:

            print("inputList[i] = ",inputList[i])
            print("inputList[i+1] = ", inputList[i+1])
            highest_element = inputList[i]
            second_highest_element = inputList[i+1]
            inputList[i],inputList[i+1] = inputList[i+1] ,inputList[i]

            print("inside if condition")
            print("Value of i = ",i)
            print("highest_element = ",highest_element)
            print("second_highest_element = ", second_highest_element)
            print("**************************")
        else:
            print("inside else condition")
            highest_element = inputList[i+1]
            second_highest_element = inputList[i]
            print("highest_element = ",highest_element)
            print("second_highest_element = ", second_highest_element)
            print("**************************")


    return second_highest_element

print(find_second_highest_number([4,3,67,54,6,98,2]))



