"""Object

"""

class Person:
    pass    # An empty block
p = Person()
print(p)
print(type(p))
print(isinstance(p, object))    # True Python 3.0以上object已经作为默认基类被继承
print(issubclass(Person, Person))
print(dir(p))

class Person1:
    def show(self):
        print('I am Person1')

# Person.__bases__ = (Person1,)    # it is a bug
# print(p.show())

class Person(object):
    def __init__(self, name):
        self.name = name
    def sayHi(self):
        print('Hello, my name is', self.name)
p = Person('Mingle')
p.sayHi()
Person('DuoDuo').sayHi()

class Robot:
    """Represents a robot, with a name."""

    # a class variable, counting the number of robots
    population = 0

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
        print('(Initialize {0})'.format(self.name))
        Robot.population += 1

    def __del__(self):
        """I am dying."""
        print('{0} is being destroyed!'.format(self.name))
        Robot.population -= 1
        if Robot.population == 0:
            print('{0} was the last one.'.format(self.name))
        else:
            print('There are still {0:d} robots working.'.format(Robot.population))

    def sayHi(self):
        """Greeting by robot.

        Yeah, they can do that."""
        print('Greetings, my master call me {0}.'.format(self.name))

    @staticmethod
    def howMany():
        """Prints the current population."""
        print('We have {0:d} robots.'.format(Robot.population))
    # howMany = staticmethod(howMany)

r1 = Robot('R1-D1')
r1.sayHi()
Robot.howMany()
r2 = Robot('R2-D2')
r2.sayHi()
Robot.howMany()
print('\nRobots can do some work here.')
print("Robots have finished their work. So let's destroy them.")
del r1
del r2
Robot.howMany()

# 继承
class SchoolMember:
    """Represent any school member."""
    def __init__(self, name, age):
        super(SchoolMember, self).__init__()
        # self.name, self.age = name, age
        self.name = name
        self.age = age
        print('(Initialize SchoolMember:{0})'.format(self.name))

    def tell(self):
        """Tell my details."""
        print("Name:'{0}' Age:'{1}'".format(self.name, self.age), end = '')
        
class Teacher(SchoolMember):
    """Repressent a teacher."""
    def __init__(self, name, age, salary):
        # SchoolMember.__init__(self, name, age)
        super(Teacher, self).__init__(name, age)
        self.salary = salary
        print('(Initialized Teacher:{0})'.format(self.name))
        
    def tell(self):
        SchoolMember.tell(self)
        print("Salary:'{0:d}'".format(self.salary))

class Student(SchoolMember):
    """Repressent a student."""
    def __init__(self, name, age, marks):
        # SchoolMember.__init__(self, name, age)
        super(Student, self).__init__(name, age)
        self.marks = marks
        print('(Initialized Teacher:{0})'.format(self.name))
        
    def tell(self):
        SchoolMember.tell(self)
        print("Marks:'{0:d}'".format(self.marks))

teacher = Teacher('Mr.Li', 30, 30000)
student = Student('Mingle', 25, 75)
print()    # pirnts a blank line
members = [teacher, student]
for member in members:
    member.tell()

