
# problem_name : best-time-to-buy-and-sell-stock
# probelem link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

def maxProfit(prices: list[int]) -> int:
    max_profit = 0

    for i in range(len(prices) - 1):
        print("**************************************")
        for j in range(i + 1, len(prices)):
            print("The value of i = ", i)
            print("The value of j = ", j)

            calculated_profit = prices[j] - prices[i]
            print("The calculated profit is ", calculated_profit)
            if calculated_profit > max_profit:
                max_profit = calculated_profit
            print("The max profit is ", max_profit)
            print("---------")

    return max_profit


prices = [7,1,5,3,6,4]
print(maxProfit(prices))