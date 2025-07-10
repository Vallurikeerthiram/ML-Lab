# Write a program to count the highest occurring character & its occurrence count in an input
# string. Consider only alphabets. Ex: for "hippopotamus" as input string, the maximally
# occurring character is 'p' & occurrence count is 3.

def freq(s):
    frequency={}
    for char in s:
        if char.isalpha():
            char=char.lower()
            frequency[char]=frequency.get(char,0)+1
    if not frequency:
        return "No alpha char in input"
    
    maxC= max(frequency,key=frequency.get)
    maccount=frequency[maxC]

    return maxC, maccount

inS= input("enter the sctring : ")
maxOccured,count=freq(inS)
print(f"The maximally occurring character is '{maxOccured}' & occurrence count is {count}.")