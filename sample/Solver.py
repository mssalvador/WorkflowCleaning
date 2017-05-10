#Dummy program, because i'm noob...

import math

class Solver:

    def demo(self,a,b,c):

        d = b**2-4*a*c

        if d>0:
            discriminant =  math.sqrt(d)

            root1 = (-b+discriminant) / (2*a)
            root2 = (-b-discriminant) / (2*a)

            return root1,root2
        elif d == 0:
            return -b/(2*a)
        else:
            return "This equation has no roots"


if __name__ == '__main__':
    solver = Solver()

    while True:
        a = int(input("a "))
        b = int(input("b "))
        c = int(input("c "))
        result = solver.demo(a,b,c)
        print(result)