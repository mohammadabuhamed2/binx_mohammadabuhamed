def MaxAndMin(numbers):
    max=numbers[0]
    min=numbers[0]
    for i in numbers:
        if i>max:
            max=i
        if i<min:
            min=i
    
    MinMax={
        "max":max,
        "Min":min
    }
    return MinMax

def EvenNumbers(numbers):
    ll=[i for i in numbers if i%2==0]
    print(ll)
    
    
l=[1,2,3,4,5,6,7,8,9]
print(MaxAndMin(l))
EvenNumbers(l)

class student:
    def __init__(self,name,id,status):
        self.name=name
        self.id=id
        self.status=status
    def IsActive(self):
        if self.status==True:
            return True
        else:
            return False
        
    
ss=student("mohammad",1212,True)
print(ss.IsActive())

        
    
    
    