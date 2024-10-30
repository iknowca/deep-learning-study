import numpy as np

def numerical_gradient(f, x):
    """
    수치적으로 함수 f의 gradient(기울기)를 계산하는 함수.
    :param f: 미분하려는 함수
    :param x: 기울기를 구하려는 점의 좌표 (numpy 배열)
    :return: 주어진 점에서의 함수 f의 기울기 (numpy 배열)
    
    각 변수에 대해 중심 차분법을 사용하여 기울기를 계산합니다.
    """
    h = 1e-4  # 아주 작은 값, 수치 미분을 위한 delta 값
    grad = np.zeros_like(x)  # 입력과 동일한 shape를 가지는 gradient 배열 초기화

    for idx in range(len(x)):  # 각 변수에 대해 반복
        tmp_val = x[idx]  # 변수의 원래 값 저장

        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh = f(x)  # 함수 f에서의 값 계산

        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)  # 함수 f에서의 값 계산

        # 중심 차분을 사용한 기울기 계산
        grad[idx] = (fxh - fxh2) / (2 * h)

        # 원래의 변수 값 복원
        x[idx] = tmp_val

    return grad

def numerical_gradient2d(f, x):
    """
    수치적으로 함수 f의 gradient(기울기)를 계산하는 함수.
    :param f: 미분하려는 함수
    :param x: 기울기를 구하려는 점의 좌표 (numpy 배열)
    :return: 주어진 점에서의 함수 f의 기울기 (numpy 배열)
    
    각 변수에 대해 중심 차분법을 사용하여 기울기를 계산합니다.
    """
    h = 1e-4  # 아주 작은 값, 수치 미분을 위한 delta 값
    grad = np.zeros_like(x)  # 입력과 동일한 shape를 가지는 gradient 배열 초기화

    # 다차원 배열일 경우 각 차원별로 수치 미분을 계산합니다.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]  # 현재 위치의 값 저장

        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)  # 함수 f에서의 값 계산

        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)  # 함수 f에서의 값 계산

        # 중심 차분을 사용한 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 원래의 변수 값 복원
        x[idx] = tmp_val
        it.iternext()

    return grad