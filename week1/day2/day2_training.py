a = [1,3,2,3,4]
print(type(a))
print(type(a[0]))
a[0]="mohammad"
print(type[a[0]])
print(type(a[0]))
a.reverse()
acopy=a.copy()
acopy.reverse()
print(a)
print(a)
print(a[1:3])
biglist=["mohammad","assem","the numbers"]
print(biglist)
biglist.sort()
print(biglist)
biglist=["mohammad","assem","the numbers",a]
print(biglist)
print("assem" in biglist)
biglist.remove("the numbers")
print(biglist)
biglist.pop(1)
print(biglist)
biglist.append("ahmaad")
print(biglist)
biglist.insert(2,"yousef")
print(biglist)
day=2
tomorow="third"
str="""its day {} of training and tomorow will be the {} day"""
finalstat=str.format(day,tomorow)
print(finalstat)
print(f"its day {day} of training and tomorow will be the {tomorow} day")
a[4]=15
a.sort(reverse=True)
print(a)
a.sort()
print(a)
print(len(a))
print(len(biglist))

t=(2,1,2)
#tuple
print(t)
print(type(t))
person1={
    "name":"mohammad",
    "age":21,
    "gender":"male",
    "address":"jenin",
    "active":True
}
print(person1["name"])
person1["name"]="ahmad"
print(person1)
st={1,2,3,4,5}
print(st)
st1=set([1,2,3,7,8,9])
print(st1)
print(1 in st1)
bigst= set({})
bigst=st|st1
print(bigst)
bigst=st&st1
print(bigst)
bigst=st-st1
print(bigst)
num1=22
if num1%2==0:
    print(f"the number {num1} is even")
else:
    print(f"the number {num1} is odd")
for i in st:
    if i==5:
        print(f"you got the number {i}")
        
        
l=[]

for i in range(50):
        if i%2==0:
            l.append(i)

print(l)

ll=[x for x in range(50) if x%2==0]
print(ll)
class datapoint:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
            
    



