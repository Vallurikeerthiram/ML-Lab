# Generate a list of 25 random numbers between 1 and 10. Find the mean, median and mode
# for these numbers.
import random
def gen(start,end,number):
    return [random.randint(start,end) for _ in range(number)]

def mean(lis):
    sum=0
    for i in lis:
        sum+=i
    return sum/len(lis)

def median(lis):
    l=sorted(lis)
    if len(l)%2==0:
        return (l[(len(l)//2) - 1] + l[(len(l)//2)])/2
    else:
        return l[len(l)//2]
    
def mode(lis):
    freq={}
    for i in lis:
        freq[i]=freq.get(i,0)+1
    maxFreq=max(freq.values())
    modes=[k for k, v in freq.items() if v == maxFreq]
    if len(modes)==1:
        return modes[0]
    else:
        return modes # to return list if multiple
    
Start=int(input("enter the start point :"))
end= int(input("enter the end point :"))
n = int(input("enter the number of numbers :"))
lis=gen(Start,end,n)
print("List is " + str(lis) + " whose mean is " + str(mean(lis)) + ", median is " + str(median(lis)) +", and mode is " + str(mode(lis)))