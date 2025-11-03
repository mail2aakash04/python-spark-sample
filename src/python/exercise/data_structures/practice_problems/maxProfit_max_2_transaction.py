def maxProfit_max_2_transactions(prices: list[int]) :
    calculated_profit = []
    for i in range(len(prices) - 1):
        for j in range(i + 1, len(prices)):
            calculated_profit.append(prices[j] - prices[i])
    return sorted(calculated_profit,reverse=True)

print(maxProfit_max_2_transactions([3,3,5,0,0,3,1,4]))

