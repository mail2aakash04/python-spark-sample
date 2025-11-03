
def two_sum_problem(nums,target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i,j]
    return []

def two_sum_2(nums, target):
    seen = {}  # number -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
        print("seen",seen)
    return []

def twoSumAakash(nums, target):


    for i, num in enumerate(nums):
        print("i = ",str(i) + " and num = " +  str(num))
        complement = target - num
        print("complement = ",complement)
        newList = nums
        newList.pop(i)
        print(newList)
        # if (complement in nums) and complement != nums[i] :
        if (complement in newList):
            print("inside if ")
            return [i, nums.index(complement)]
        print("***********",nums)
    return  None


# two_sum_2([2,11,7,15],9)  # [0,1]

# print(twoSumAakash([22,2,11,15,7],17))  # [0,1]
print(twoSumAakash([3,2,4],6))
# print(two_sum_problem([2,7,11,15],9))  # [0,1]
