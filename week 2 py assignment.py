# Create an empty list called my_list
my_list = []

# Append the elements 10, 20, 30, 40
my_list.append(10)
my_list.append(20)
my_list.append(30)
my_list.append(40)
print("After appending 10, 20, 30, 40:", my_list)
# Output: After appending 10, 20, 30, 40: [10, 20, 30, 40]

# Insert the value 15 at the second position (index 1)
my_list.insert(1, 15)
print("After inserting 15 at index 1:", my_list)
# Output: After inserting 15 at index 1: [10, 15, 20, 30, 40]

# Extend my_list with another list [50, 60, 70]
my_list.extend([50, 60, 70])
print("After extending with [50, 60, 70]:", my_list)
# Output: After extending with [50, 60, 70]: [10, 15, 20, 30, 40, 50, 60, 70]

# Remove the last element from my_list
my_list.pop()
print("After removing the last element:", my_list)
# Output: After removing the last element: [10, 15, 20, 30, 40, 50, 60]

# Sort my_list in ascending order
my_list.sort()
print("After sorting in ascending order:", my_list)
# Output: After sorting in ascending order: [10, 15, 20, 30, 40, 50, 60]

# Find and print the index of the value 30
index_of_30 = my_list.index(30)
print("Index of value 30:", index_of_30)
# Output: Index of value 30: 3
