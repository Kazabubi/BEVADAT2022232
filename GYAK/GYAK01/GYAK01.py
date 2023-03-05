def contains_odd(numlist = []):
    for number in numlist:
        if number % 2 == 1:
            return True
    return False

def is_odd(numlist = []):
    boollist = []
    for number in numlist:
        if number % 2 == 1:
            boollist.append(True)
        else:
            boollist.append(False)
    return boollist

def element_wise_sum(alist = [], blist = []):
    sumlist = []
    for index in range(0, len(alist)):
        if index < len(blist):
            sumlist.append(alist[index] + blist[index])
    return sumlist

def dict_to_list(dict = {}):
    dictlist = []
    for key, value in dict.items():
        dictlist.append((key, value))
    return dictlist

def main():
    #print(contains_odd([0, 2, 6, 7, 8]))  
    #print(is_odd([0, 2, 6, 7,8]))
    #print(element_wise_sum([1, 2, 3, 4, 78], [1, 2, 3, 4, 7, 9, 43]))
    print(dict_to_list({"egy":1,"ketto":2,"harom":3}))

main()
