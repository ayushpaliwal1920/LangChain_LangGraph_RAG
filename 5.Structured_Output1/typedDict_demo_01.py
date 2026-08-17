from typing import _TypedDict

class Person(_TypedDict):
    name : str
    age : int

new_person : Person = {
     'name' : 'nitish',
     'age' : 35
}

print(new_person)