# Write a program that takes a list of real numbers as input and returns the range (difference between minimum and maximum) of the list. Check for list being less than 3 
# elements in which case return an error message (Ex: "Range determination not possible"). Given a list [5,3,8,1,0,4], the range is 8 (8-0).

def rangeOflist(li):
    if len(li)<3:
        raise ValueError("Range determination not possible")
    else:
        return [max(li),min(li)]

arr= list(map(int, input("enter numbers seperated with spaces: ").split()))
print("Given a list " + str(arr) + ", the range is " +str((rangeOflist(arr))[0] - (rangeOflist(arr))[1]) + " (" +str((rangeOflist(arr))[0]) + " - " + str((rangeOflist(arr))[1]) + ")")