def subset(start, end, numlist = []):
    sublist = []
    for index in range(start, end):
        sublist.append(numlist[index])
    return sublist

def every_nth(n, numlist = []):
    sublist = []
    for index in range(n):
        sublist.append(numlist[index])
    return sublist

def unique(numlist = []):
    numlist.sort()
    for index in range(1, len(numlist)):
        if numlist[index-1] == numlist[index]:
            return False
    return True

def flatten(matrix = [[]]):
    flatlist = []
    for instance in matrix:
        for item in instance:
            flatlist.append(item)
    return flatlist

def merge_lists(*matrix):
    flatlist = []
    for instance in matrix:
        flatlist = flatlist + instance
    return flatlist

def reverse_tuples(tuplist = [()]):
    for index in range(len(tuplist)):
        tuplist[index] = (tuplist[index][1], tuplist[index][0])
    return tuplist

def remove_duplicates(numlist = []):
    numlist.sort()
    index = 0
    while index != len(numlist)-1:
        if numlist[index] != numlist[index +1]:
            index = index + 1
        else:
            numlist.remove(index+1)
    return numlist

def transpose(matrix = [[]]):
    transposed = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transposed.append(row)
    return transposed

def split_into_chunks(chunk_size, numlist = []):
    chunklist = []
    chunk = []
    fragmentcounter = 0
    for item in numlist:
        if fragmentcounter == chunk_size:
            fragmentcounter = 0
            chunklist.append(chunk)
            chunk = []
        chunk.append(item)
        fragmentcounter += 1
    chunklist.append(chunk)
    return chunklist

def merge_dicts(*dictionaries):
    mergeddict = {}
    for instance in dictionaries:
        for key, value in instance.items():
            mergeddict[key] = value
    return mergeddict

def by_parity(numlist):
    paritydict = {}
    paritydict["even"] = []
    paritydict["odd"] = []
    for item in numlist:
        if item % 2 == 0:
            paritydict["even"].append(item)
        else:
            paritydict["odd"].append(item)
    return paritydict

def mean_key_value(numdict = {}):
    meandict = {}
    for key, value in numdict.items():
        numsum = 0
        counter = 0
        for item in value:
            numsum += item
            counter += 1
        meandict[key] = numsum / counter
    return meandict
        


def main():
    print(subset(1, 3, [1,4,3,2,5]))
    print(every_nth(3, [1,4,3,2,5]))
    print(unique([1,4,3,2,5, 4]))
    print(flatten([[1,2],[3,4],[5,6]]))
    print(merge_lists([1,2],[3,4],[5,6]))
    print(reverse_tuples([(1,2),(3,4),(5,6)]))
    print(remove_duplicates([1,4,3,2,5, 4]))
    print(transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
    print(split_into_chunks(3, [1,2,3,4,5,6,7,8]))
    print(merge_dicts({"one":1,"two":2}, {"four":4,"three":3}))
    print(by_parity([1,2,3,4,5,6]))
    print(mean_key_value({"some_key":[1,2,3,4],"another_key":[1,2,3,4]}))

main()
