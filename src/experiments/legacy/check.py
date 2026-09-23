r=''
def foo():
    global r
    r='hello'
def koo():
    global r
    r='ghhello'
def goo():
    
    foo()
    koo()
    print(r)

goo()
