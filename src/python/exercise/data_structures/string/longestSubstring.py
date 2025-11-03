from collections import Counter

def lengthOfLongestSubstring( s: str) -> int:
        counter = 0
        outputList = []
        substringList = []
        for i,num in enumerate(s):
            print("***********************")
            if s[i] in substringList:
                print("inside if condition")
                counter = 0
                substringList = substringList[1:]
                counter = len(substringList)
                outputList.append(counter)
                substringList += s[i]
                print("counter = ", str(counter) + " and substringList = " + str(substringList))
            else:
                substringList += s[i]
                counter = len(substringList)
                print("counter = ", str(counter) + " and substringList = " + str(substringList))
        outputList.append(counter)
        print(outputList)
        return max(outputList)

print(lengthOfLongestSubstring("qwertqwertyu"))
