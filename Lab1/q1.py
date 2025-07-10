"Consider the given list as [2, 7, 4, 1, 3, 6]. Write a program to count pairs of elements with sum equal to 10."

def count_pairs(list, tar):
    checked = set() #checked pairs
    result = set() 

    for i in list:
        dif = tar - i 
        if dif in checked:
            result.add(tuple(sorted([i,dif])))
        checked.add(i)

    return len(result)

arr= list(map(int, input("enter numbers seperated with spaces: ").split()))
tar= int(input("enter the target number: "))
print(count_pairs(arr,tar))
