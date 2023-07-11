#!/usr/bin/env python
# coding: utf-8

# In[3]:


x = ['it']
y = [ 'th']
print(x+y)
import numpy as np
p = np.array(x)
q = np.array(y)
r = np.char.add(p,q)
print(r)


# In[2]:


x = [1,2,3]
y = [4,5,6]
print(x+y)
import numpy as np
p = np.array(x)
q = np.array(y)
print(p+q)


# In[3]:


light = p>2
print(light)
print(p[light])


# In[59]:


import random

def create_random_bmi(size):
    bmi = []

    for _ in range(size) :
        weight = random.uniform(40, 100)
        height = random.uniform(1.5,2.0)
        b = weight/(height**2)
        bmi.append(b)
    return bmi
bk = create_random_bmi(10)
bq = create_random_bmi(5)
bk_rounded = np.round(bk, 2)
print(bk_rounded)

bq_rounded = np.round(bq, 2)
print(bq_rounded)


# In[30]:


light = bmi >12
print(light)
print(bmi[light])


# In[36]:


print(np.mean(bmi))
print(np.median(bmi))


# In[4]:


population = [1000000, 500000, 700000, 300000, 900000]
gdp = [25000000000, 15000000000, 18000000000, 9000000000, 20000000000]
s = [3,5,6,7,80]
import matplotlib.pyplot as plt
plt.xlabel('population')
plt.ylabel('gdp')
plt.title('Ppulation vs GDP')
plt.xticks([0.4*10**6,0.8*10**6,1.2*10**6])
plt.yticks([0.5*10**10,1.0*10**10,1.5*10**10,2.0*10**10,2.5*10**10])
plt.scatter(population, gdp, s, c = "red", alpha = 0.5)
plt.grid(True)
plt.show()

plt.clf()

population = [1000000, 500000, 700000, 300000, 900000]
gdp = [25000000000, 15000000000, 18000000000, 9000000000, 20000000000]


plt.scatter(population, gdp, s = [3,5,6,7,80], c = "purple", alpha = 0.9)
plt.grid(False)
plt.show()


# In[3]:


brics = {
    'Brazil': [211_049_527, 8_515_767, 3_365_992_000_000, 'South America'],
    'Russia': [145_912_025, 17_098_242, 1_699_889_000_000, 'Europe/Asia'],
    'India': [1_366_417_754, 3_287_263, 2_975_143_000_000, 'Asia'],
    'China': [1_409_517_397, 9_596_961, 14_342_903_000_000, 'Asia'],
    'South Africa': [58_558_270, 1_221_037, 351_431_000_000, 'Africa']
}

print(brics['India'][0])


# In[4]:


brics['India'][0] = 1366417760
print(brics)


# In[5]:


print(brics.keys())


# In[6]:


print(brics['Brazil'])


# In[7]:


import pandas as pd
pd.DataFrame(brics)


# In[8]:


brics.index = ["Pop","Area","GDP","Continent"]


# In[ ]:


data = {
    'Country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'Population': [211_049_527, 145_912_025, 1_366_417_754, 1_409_517_397, 58_558_270],
    'Area': [8_515_767, 17_098_242, 3_287_263, 9_596_961, 1_221_037],
    'GDP': [3_365_992_000_000, 1_699_889_000_000, 2_975_143_000_000, 14_342_903_000_000, 351_431_000_000],
    'Continent': ['South America', 'Europe/Asia', 'Asia', 'Asia', 'Africa']
}


# In[9]:


brics = pd.DataFrame(data)
print(brics)


# In[10]:


brics.index = ['BR', 'RU', 'IN', 'CH', 'SA']
print(brics)


# In[11]:


print(brics[['Country', 'Area']])
print(brics[0:2])


# In[12]:


print(brics.loc[['BR','SA'],['Country', 'Area']])


# In[13]:


print(brics.iloc[[0,4],[0,2]])


# In[14]:


import random
import numpy as np
def create_random_dict(size):
    dict = {}
    for _ in range(size):
        keys = random.randint(1,100)
        value_pairs = random.randint(1,1000)
        dict[keys] = value_pairs
        
    return(dict)
dk = create_random_dict(8)
print(dk)


# In[15]:


import random 
import numpy as np
def create_random_bmi(size):
    bmi = []
    for _ in range(size):
        weight = random.uniform(40,100)
        height = random.uniform(1.5,2)
        b = weight/height**2
        bmi.append(b)
    return(bmi)
dns = np.round((create_random_bmi(5)), 2)
print(dns)


# In[16]:


print(brics['Population']>145912025)


# In[17]:


print(brics[brics['Population']>145912025])


# In[18]:


import numpy as np
print(np.logical_not(brics['Population'] > 145912025, brics['Population'] < 1366417754))


# In[19]:


print(brics[np.logical_and(brics['Population'] > 145912025, brics['Population'] < 1366417754)])


# In[20]:


bool(5)+7


# In[21]:


'Python'+'sol'


# In[22]:


True + 'data'


# In[33]:


def greet(name, message = 'hibro'):
    print(message, name)


greet('Manogna')


# In[39]:


def tyty(p, q, w=None):
    print(p*q)
    print(p*w)
tyty(1,2)


# In[40]:


'ty' *2


# In[47]:


x = ['it', 'yu']
print(x * 2)
import numpy as np
p = np.array(x)
print(p * 2)


# In[49]:


x = ['it', 'yu']
print(x * 2)
import numpy as np
p = np.array(x)

print(np.char.multiply(p,2))


# In[66]:


x = ['it', 'yu']
y = ['as', 'mk']
o = [1]
t = [5]
print(x * 2)
import numpy as np
p = np.array(x)
q = np.array(y)


print(np.char.multiply(p, q))


# In[55]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5])
multiplied = arr * 2

print(multiplied)


# In[59]:




import numpy as np
p = np.array(['it', 'yu'])
print(p * 2)


# In[60]:


[1, 2, 3, 4, 5] * 2


# In[62]:


(1, 2, 3, 4, 5) *2


# In[71]:


import numpy as np

x = np.array(['it', 'yu'])
y = np.array([1, 2])

result = np.char.multiply(x, y)
print(result)


# In[76]:


'ty'+'ty'

'ty'*'ty'


# In[75]:


x = ['it', 'yu']
y = ['as', 'mk']
print(x+y)


# In[84]:


import numpy as np

x = np.array(['s, 2])
print(np.char.add(x,y))


# In[85]:


x = [1]
y = [2]
print(x+y)


# In[90]:


fruits = ['apple', 'banana', 'orange']

for index, f in enumerate(fruits):
    print("Index: " + str(index)+', '+'Fruit: '+ str(f))






# In[ ]:


Index: 0, Fruit: apple


# In[3]:


import numpy as np
p = np.array(['rt','bg'])
q = np.array(['ir', 'ui'])
print(np.char.add(p,q))


# In[22]:


dict = {}
dict['id'] = 'west'
dict['gh'] = 'north'
dict['li'] = 'south'
dict['iy'] = 'east'


for key, value in dict:
    print(key,value)


# In[38]:


import numpy as np
s = np.array([1,2,3,5])
u = np.array([4,6,7, 9])
p = np.array([s, u])

for a in np.nditer(p):
    print(a)


# In[33]:


import numpy as np

p = np.array([[1, 2, 3, 5],
              [4, 6, 7, 9]])

# Using a loop to print each value individually
for value in p.flatten():
    print(value)
    


# In[39]:


import numpy as np

# Example arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Concatenate along the 0th axis (default)
concatenated_array = np.concatenate(array1, array2)

print(concatenated_array)


# In[86]:


import numpy as np

# Example arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.hstack((array1, array2))
array4 = np.array([17, 26, 34])
array5 = np.array([7, 89, 65])
array6 = np.hstack((array4, array5))
array7 = np.vstack((array3, array6))
print(array3)
print(np.vstack((array3, array6)))


# In[61]:


import numpy as np

p = np.array([[1, 2],
              [3, 4],
              [5, 8]])

q = np.array([[5, 6],
              [7, 8],
              [9, 10]])

# Using np.concatenate() to concatenate along the 0th axis
result_concatenate = np.concatenate((p, q), axis=0)
print("Concatenate:")
print(result_concatenate)
print()

# Using np.hstack() to horizontally stack arrays
result_hstack = np.hstack((p, q))
print("Horizontal Stack:")
print(result_hstack)

result_vstack = np.vstack((p, q))
print("Vertical Stack:")
print(result_vstack)


# In[64]:


brics = {
    'Brazil': 'Brasilia',
    'Russia': 'Moscow',
    'India': 'New Delhi',
    'China': 'Beijing',
    'South Africa': 'Pretoria/Cape Town',
}

for key, value in brics.items():
    print(key +'--' + value)


# In[67]:


import numpy as np

# Example BMI arrays
bmi_array1 = np.array([22.5, 27.8, 19.6, 31.2, 25.4])
bmi_array2 = np.array([18.9, 26.1, 23.7, 29.5, 21.3])

b = np.array([bmi_array1, bmi_array2])

for i in np.nditer(b):
    print(i)


# In[68]:


import numpy as np

# Example BMI arrays
bmi_array1 = np.array([22.5, 27.8, 19.6, 31.2, 25.4])
bmi_array2 = np.array([18.9, 26.1, 23.7, 29.5, 21.3])

b = np.array([bmi_array1, bmi_array2])

for i in b.flat:
    print(i)


# In[69]:


import numpy as np

# Example BMI arrays
bmi_array1 = np.array([22.5, 27.8, 19.6, 31.2, 25.4])
bmi_array2 = np.array([18.9, 26.1, 23.7, 29.5, 21.3])

b = np.array([bmi_array1, bmi_array2])

for i in b.flatten():
    print(i)


# In[87]:


import numpy as np

# Example BMI arrays
bmi_array1 = np.array([[22.5, 27.8, 19.6, 31.2, 25.4], [18.9, 26.1, 23.7, 29.5, 21.3]])
bmi_array2 = np.array([[18.9, 26.1, 23.7, 29.5, 21.3], [22.5, 27.8, 19.6, 31.2, 25.4]])

b = np.vstack((bmi_array1, bmi_array2))
print(b)
alex = b.flat

print(list(alex))


# In[77]:


import numpy as np

# Example BMI arrays
bmi_array1 = np.array([22.5, 27.8, 19.6, 31.2, 25.4])
bmi_array2 = np.array([18.9, 26.1, 23.7, 29.5, 21.3])

b = np.vstack((bmi_array1, bmi_array2))
alex = b.flatten()

print(alex)


# In[90]:


import numpy as np
bmi_array1 = np.array([22.5, 27.8, 19.6, 31.2, 25.4])
bmi_array2 = np.array([18.9, 26.1, 23.7, 29.5, 21.3])

light = np.logical_and(bmi_array1 > 25, bmi_array1 < 30)
print(light)
print(bmi_array1[light])


# In[107]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

import pandas as pd



brics = pd.DataFrame(brics_data)
brics.index = ['b', 'r', 'i', 'c', 's']
print(brics)
print(brics['countries'][0])
print(brics[1:2])




# In[113]:


import numpy as np

# Create the first 2x2 array
array1 = np.array([[1, 2], [3, 4]])

# Create the second 2x2 array
array2 = np.array([[5, 6], [7, 8]])

array3 = np.concatenate((array1, array2))
# Print the arrays

print(array3)

print(list(array3.flat))
print(array3.flatten())


# In[120]:


import numpy as np
a = [1, 2, 3, 4, 5, 6, 7, 8]

for index, r in enumerate(a):
    print(index, r)


# In[121]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

for key, value in brics_data.items():
    print(key, value)


# In[123]:


import numpy as np

# Create the first 2x2 array
array1 = np.array([[1, 2], [3, 4]])

# Create the second 2x2 array
array2 = np.array([[5, 6], [7, 8]])

array3 = np.array([array1, array2])
print(array3)

for a in np.nditer(array3):
    print(a)


# In[129]:


import numpy as np

# Create the first 2x2 array
array1 = np.array([[1, 2], [3, 4]])

# Create the second 2x2 array
array2 = np.array([[5, 6], [7, 8]])

array3 = np.array([array1, array2])

print('Python_List - ' + str(list(array3.flat)))
print('Numpy_Array - ' + str(array3.flatten()))
print(np.vstack((array1, array2)))
print(np.hstack((array1, array2)))
print(np.concatenate((array1, array2)))


# In[137]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

import pandas as pd
bricsc = pd.DataFrame(brics_data)

print(type(brics[['countries', 'gdp']]))
print(type(brics['countries']))


# In[161]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

import pandas as pd
brics = pd.DataFrame(brics_data)
brics.index = ['b', 'r', 'i', 'c', 's']


print(brics.loc[['b', 'i'], ['countries', 'gdp']])


# In[167]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

for i, k  in brics_data.items():
    print(i)


# In[176]:


import numpy as np

# Create the first 2x2 array
array1 = np.array([[1, 2], [3, 4]])

# Create the second 2x2 array
array2 = np.array([[5, 6], [7, 8]])

array3 = np.array([array1, array2])


for i in np.nditer(array3):
    print(i)
    
print(array3.flatten())
print(list(array3.flat))


# In[179]:


import numpy as np
array1 = np.array([1, 2, 3])
for i in array1:
    print(i)                  


# In[191]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

import pandas as pd
brics = pd.DataFrame(brics_data)
brics.index = ['b', 'r', 'i', 'c', 's']
for i, k in brics.iterrows():
    brics.loc[[i], ['coun_len']] = len(k['countries'])
print(brics)
    


# In[193]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [211049519, 145912025, 1400059927, 1439323776, 58775022],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}

import pandas as pd
brics = pd.DataFrame(brics_data)
brics.index = ['b', 'r', 'i', 'c', 's']

brics['coun_len'] = brics['countries'].apply(len)
print(brics)


# In[201]:


import random
def bmi_gen(size):
    bmi = []
   
   
    for _ in range(size):
        height = random.uniform(1.5, 2)
        weight = random.uniform(40, 120)
        b = weight/height**2
        bmi.append(b)
       
    
    return(bmi)
bmi_gen(5)


# In[212]:


import random
random.seed(42)
def dict(size):
    dict = {}
    for _ in range(size):
        key = random.randint(1,100)
        value1 = random.randint(1,1000)
        value2 = random.randint(1000, 10000)
        dict[key] = [value1, value2]
    return(dict)
dict(10)


# In[215]:


import random
random.seed(42)
coin = random.randint(0,2)
if (coin == 0):
    print('Head')
else:
    print('Tail')


# In[226]:


import numpy
import random
random.seed(123)
tails = [0]

for x in range(10):
    coin = random.randint(0,1)
    print(coin)
    print(tails[x])
    print('-------')
    tails.append(tails[x] + coin)
print(tails)
    


# In[235]:


import numpy as np
step = [0]

for i in range(100):
    x = random.randint(1,6)
    print(x)
    if x<=2:
        p = max(0, step[i]-1)
    elif x<=5:
        p = step[i]+1
    else:
        p = step[i] + np.random.randint(1,6)
    step.append(p)
print(step)
print(step[-1])


# In[64]:



import numpy as np
import random
import matplotlib.pyplot as plt

def do_it(size):
    final = []
    for _ in range(size):     
        sp = [0]
        
        for i in range(100):
            dice = random.randint(1,6)
            step = sp[i]
            if dice <= 2:
                st = max(0, step - 1)
            elif dice <=5:
                st = step + 1
            else:
                st = step + random.randint(1,6)
            sp.append(st)
        fi = sp[-1]
        final.append(fi)
        
       
    return(np.mean(final))    
do_it(100)


# In[29]:


print(sp)
plt.plot(sp)
plt.xlabel('step position')
plt.xticks([15,30,45,60,75,90,105],['15th','30th','45th','60th','75th','90th','105th'])
plt.show()


# In[4]:


import numpy as np
import random
import matplotlib.pyplot as plt

def do_it(size):
    final = []
    for _ in range(size):     
        sp = [0]
        
        for i in range(100):
            dice = random.randint(1,6)
            step = sp[i]
            if dice <= 2:
                st = max(0, step - 1)
            elif dice <=5:
                st = step + 1
            else:
                st = step + random.randint(1,6)
            sp.append(st)
        fi = sp[-1]
        final.append(fi)
        plt.xticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140])
        plt.hist(final, color='red')
       
    return(plt.show())    
do_it(1000)


# In[83]:


import numpy as np
import matplotlib.pyplot as plt
x = np.array([[1,2,3,4], [2,3,4,6]])
y = np.transpose(x)
plt.plot(x)
plt.show()
plt.clf()
plt.plot(y)
plt.show()
print(x)
print(y)
print(x[-1][0])


# In[78]:


from numpy import random
x = random.rand()
print(x)


# In[105]:


import numpy as np
import matplotlib.pyplot as plt
heights = np.array([16, 172, 57, 250, 100])
populations = np.array([1000000, 500000, 1500000, 800000, 1200000])
plt.xticks([500000, 1000000, 1500000, 20000000], ['0.5 M', '1 M', '1.5 M', '2 M'])
plt.scatter(populations,heights, s = heights, c = 'red', alpha = 0.5)
plt.show()


# In[91]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Graph')
plt.show()


# In[120]:


import numpy as np
my_list =np.array([[1, 2], [3, 4]])
for j, i in enumerate(np.nditer(my_list)):
    print( j, i)


# In[3]:


import numpy as np
my_list =np.array([[1, 2], [3, 4]])
print(my_list[0][1])


# In[8]:


h = [11,2,3,45,87,39,6,23,5,67]
q = [i>40 and i<70 for i in h]
print(q)
j = [k for k in h if k>40 and k<70]
print(j)


# In[13]:


manu = [10,20,30,40,50,60,70,80,90,100,110,120,130,140]
 
print([k>50 and k<111 for k in manu])
print([k for k in manu if k>50 and k<111])


# In[19]:


import numpy as np
mamu = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140])
q = np.logical_and(mamu>50, mamu<111)
mamu[q]


# In[20]:


import numpy as np
mamu = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140])
q = mamu>50 and mamu<111
mamu[q]


# In[34]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
print(bb)
print(bb.head(3))
print(bb.describe())
print(bb.columns)
print(bb.index)
print(bb.values)
print(bb.info())
print(bb.shape)


# In[36]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)

print(bb.head(2))
print(bb.info())
print(bb.columns)
print(bb.index)
print(bb.describe())
print(bb.shape)
print(bb.values)


# In[39]:


x = [1,3,5,8]
for index, i in enumerate(x):
    print(index, i)
    


# In[57]:


import numpy as np
y = np.array([[1,3,5,8], [3,5,6,7]])
print(y)

for index, i in enumerate(y):
#     print(i)
    print(index, i)


# In[69]:


import numpy as np
a = np.array([[1,3,5,8]])
b = np.array([[3,5,6,7]])
print(a)
print(b)
c = np.concatenate((a, b), axis=0)
print(c)
d = np.hstack((a, b))
e = np.vstack((a, b))
print(d)
print(e)


# In[66]:


import numpy as np
a = np.array([[1,3,5,8], [2,7,4,0]])
b = np.array([[3,5,6,7], [6,3,1,9]])
print(a)
print(b)
print(a + b)
c = np.concatenate((a, b), axis=0)
print(c)
d = np.hstack((a, b))
e = np.vstack((a, b))
print(d)
print(e)


# In[70]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
for i in data:
    print(i)
    


# In[71]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
for i in data:
    print (i, data[i])


# In[72]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
for key, value in data.items():
    print (key , value)


# In[88]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [2110, 1459, 14000, 14390, 5877],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}
import numpy as np
import pandas as pd
brics = pd.DataFrame(brics_data)
print(brics)
print(brics['population'][np.logical_and(brics['population']>1500, brics['population']<14200)])


# In[97]:


brics_data = {
    'countries': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'population': [2110, 1459, 14000, 14390, 5877],
    'area': [8515767, 17098242, 3287590, 9706961, 1221037],
    'gdp': [1.449, 1.464, 2.971, 14.342, 0.358]
}
import numpy as np
import pandas as pd
brics = pd.DataFrame(brics_data)
brics.index = ['BR','RU' ,'IN', 'CH', 'SA']
brics


# In[102]:


# for i in brics:
#     print(i)
for lab, row in brics.iterrows():
    print(lab)
#     print('---------')
    print(row)
    print('---------')
    print('---------')


# In[104]:


for lab, row in brics.iterrows():
    brics.loc[['lab'],['cap_con']] = brics.loc[['lab']['countries']].isupper()
print(brics)


# In[107]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
print(bb.head(3))
print(bb.info())
print(bb.describe())
print(bb.shape)
print(bb.columns)
print(bb.index)
print(bb.values)


# In[112]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)

print(bb.iloc[[2], [2, 1]])


# In[131]:


ss = np.array([0.12, 0.134, 0.34, 0.4545, 0.7000000000000001, nan, 0.343])
man = []
for i in ss:
    x = len(str(i)) - 2
    
    k = i*(10**x)
    man.append(int(k))
print(man)


# In[130]:


ss = np.array([0.12, 0.134, 0.34, 0.4545, 0.7000000000000001, 0.343])
man = []

for i in ss:
    y = str(i)
    z = y[2:]
    man.append(int(z))
    
print(man)
        


# In[135]:


import numpy as np
np.array([1,2,3, np.nan])


# In[18]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
for lab, row in bb.iterrows():
    bb.loc[[lab], ['len_Name']]= len(row['Name'])
print(bb)


# In[22]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
bb['len_Name'] = bb['Name'].apply(len)
print(bb)


# In[19]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
for lab, row in bb.iterrows():
    bb.loc[[lab], ['up_Name']]= row['Name'].upper()
print(bb)


# In[24]:


data = {'Name': ['John', 'Alice', 'Bob', 'Emily', 'David'],
        'Age': [25, 28, 32, 30, 27],
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'USA']}
import numpy as np
import pandas as pd
bb = pd.DataFrame(data)
bb['up_Name'] = bb['Name'].apply(str.upper)
print(bb)


# In[32]:


import random
random.seed(1234)
def t(size):    
    tails = 0
    for _ in range(10):

        x = random.randint(0,1)
        if x == 1:
            tails = tails + 1
    return(tails)
t(10)


# In[ ]:


import numpy as np
x = np.array([1,3,5,6], [4,5,7,9])


# In[63]:


dd = {
    'name': ['Buddy', 'Max', 'Lucy', 'Charlie', 'Cooper', 'Rocky', 'Daisy'],
    'breed': ['Labrador Retriever', 'German Shepherd', 'Golden Retriever', 'Bulldog', 'Poodle', 'Boxer', 'Beagle'],
    'color': ['Yellow', 'Black and Tan', 'Golden', 'Brindle', 'Apricot', 'Fawn', 'Tri-color'],
    'date_of_birth': ['2015-06-10', '2017-02-25', '2016-09-12', '2018-04-05', '2014-11-20', '2013-08-15', '2019-01-02'],
    'height': [24, 26, 22, 18, 20, 23, 16],
    'weight': [70, 85, 65, 50, 45, 75, 30]
}
import pandas as pd
dog = pd.DataFrame(dd) 
print(dog)
print(dog.sort_values('height', ascending = True, inplace = False))
print('---------------------------------------')
print(dog[dog['breed'] == 'Poodle'])

is_Poodle = dog['breed'] == 'Poodle'
is_Apricot = dog['color'] == 'Apricot'
dog[is_Poodle & is_Apricot]

is_poo_or_bull = dog['breed'].isin(['Poodle', 'Bulldog'])
print(dog[is_poo_or_bull])


# In[80]:


import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)

step_list_of_lists = []
for j in range(100):
    step = [0]
    for i in range(99):

        dice = random.randint(1,6)
        st = step[i]
        if dice < 3:
            st = max(0, st - 1)
        elif dice < 6:
            st = st + 1
        else:
            st = st + random.randint(1,6)

        step.append(st)
        
    
    step_list_of_lists.append(step)


y = np.transpose(step_list_of_lists)
plt.figure(figsize = (30, 30))
plt.plot(y, marker = 'o')
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[74]:


import random
print(random.randint(1,2))


# In[ ]:




