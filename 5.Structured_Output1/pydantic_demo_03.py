from pydantic import BaseModel , EmailStr , Field
from typing import Optional

class Student(BaseModel) : 

    name : str = 'AYUSH'        # Default
    age  : Optional[int] = None  # Optional   
    email : EmailStr            # BUiltIn validation
    cgpa  : float = Field(gt = 0 , lt= 10 , default= 5 , description= " Representation of academic strength") # Field fxn : uses for constraints , description etc .



new_student = {"name" : "nitish" ,"age" : 35 , "email" : "abc@gmail.com", "cgpa" : 5.6 }
new_student2 = {"name" : 25 ,"age" : 35}
new_student3 = {}



student = Student(**new_student)
student2 = Student(**new_student2)
student3 = Student(**new_student3)

student_dict = dict(student) # cconvert to json / dict
student_json = student.model_dump_json() # cconvert to json / dict

print(student_dict['age'])

print(student)  # General

print(type(student))

# print(student2) # invalid

# print(student3) # Default


