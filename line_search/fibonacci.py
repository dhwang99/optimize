#econding: utf8

ll=[1,1]

for i in range(3, 50):
    n = ll[-1] + ll[-2]
    ll.append(n)

print ll


