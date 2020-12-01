import numpy as np


if __name__ == '__main__':
    print("명제")
    print(not(True and True or False))
    print(bool(10 < 20 and 0))
    print(bool(False or 1))
    print(None and None)
    print(0 or 1)
    print(bool(1) and bool(1))
    print(0 and 0)
    print()
    print()

    print("함수")
    print("-10 <= input <= 10을 만족하는 실수 입력 받는 함수 f(input)를 새롭게 정의하고, "
          "다음 명제 p가 참이 되도록 하시오.")
    print("input의 값을 초기화하거나 새로운 변수를 반환할 수 없음.")
    print("p: 0 <= f(real_number) <= 1")

    def f(real_number):
        return (real_number - (-10)) / (10 - (-10))

    done = True
    arange = np.arange(-10.0, 10.0, 0.1)
    for r in arange:
        if 0 <= f(r) <= 1:
            done = True
        else:
            done = False
        if not done:
            print("반례가 존재함.")
            break
    if done:
        print("명제는 참이다.")
    print()
    print()

    print("클래스")
    print("다음의 조건들을 만족하는 클래스를 정의하시오.")
    print("1. 부모 클래스 1개, 자식 클래스 1개")
    print("2. 부모 클래스의 함수는 2개이며 자식 클래스의 함수 역시 2개")
    print("3. 부모 클래스의 함수 이름과 자식 클래스의 함수 이름은 모두 동일")
    print("4. 부모 클래스의 첫 번째 함수는 입력이 하나이며 입력이 만약 홀수 일때 True 반환")
    print("5. 부모 클래스의 두 번째 함수는 입력이 하나이며 입력이 만약 짝수 일때 True 반환")
    print("6. 자식 클래스의 첫 번째 함수는 객체 super()가 활용되어야하며 입력이 만약 홀수라면 입력된 값을 반환")
    print("7. 자식 클래스의 두 번째 함수 역시 객체 super()가 활용되어야하며 입력이 만약 짝수라면 입력된 값을 반환")

    class Classifier:
        def odd_clf(self, _input):
            if type(_input) is int:
                if _input % 2 == 1:
                    return True

        def even_clf(self, _input):
            if type(_input) is int:
                if _input % 2 == 0:
                    return True


    class ClassifierSon(Classifier):
        def odd_clf(self, _input):
            if super().odd_clf(_input):
                return _input

        def even_clf(self, _input):
            if super().even_clf(_input):
                return _input

    par = Classifier()
    son = ClassifierSon()
    print(par.odd_clf(1))
    print(par.odd_clf(2))
    print(son.odd_clf(1))
    print(son.odd_clf(2))
    print(par.even_clf(1))
    print(par.even_clf(2))
    print(son.even_clf(1))
    print(son.even_clf(2))
