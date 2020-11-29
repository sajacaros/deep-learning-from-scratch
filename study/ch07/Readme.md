### CHAPTER 7 합셩곱 신경망(CNN)
#### 7.1 전체 구조
* Affine 계층으로 이뤄진 네트워크
    ![Affine 계층](../images/fig_7-1.png)
* CNN으로 이뤄진 네트워크
    ![CNN](../images/fig_7-2.png)
    - 합셩곱 계층(convolutional layer)
    - 폴링 계층(pooling layer)
#### 7.2 합성곱 계층
##### 7.2.1 완전연결 게층의 문제점
##### 7.2.2 합성곱 연산
    ![convolution](../images/fig_7-3.png)
    ![convolution compute](../images/fig_7-4.png)
    ![convolution bias](../images/fig_7-5.png)
##### 7.2.3 패딩
    ![padding](../images/fig_7-6.png)
##### 7.2.4 스트라이드
    ![stride](../images/fig_7-7.png)
    ![stride formal](../images/e_7-1.png)
##### 7.2.5 3차원 데이터의 합성곱 연산
    ![3dimension](../images/fig_7-8.png)
    ![3dimension order](../images/fig_7-9.png)
##### 7.2.6 블록으로 생각하기
    ![block](../images/fig_7-10.png)
    ![block with filter](../images/fig_7-11.png)
    ![block with bias](../images/fig_7-12.png)
##### 7.2.7 배치 처리
    ![batch](../images/fig_7-13.png)

#### 7.3 풀링 계층
    ![pooling order](../images/fig_7-14.png)
##### 7.3.1 풀링 계층의 특징
    ![pooling feature 1](../images/fig_7-15.png)
    ![pooling feature 2](../images/fig_7-16.png)
#### 7.4 합성곱/풀링 계층 구현하기
##### 7.4.1 4차원 배열
##### 7.4.2 im2col로 데이터 전개하기
    ![im2col](../images/fig_7-17.png)
    ![filter](../images/fig_7-18.png)
    ![reshape](../images/fig_7-19.png)
##### 7.4.3 합성곱 계층 구현하기
    ![transpose](../images/fig_7-20.png)
##### 7.4.4 풀링 계층 구현하기
    ![pooling 2-2](../images/fig_7-21.png)
    ![flow pooling](../images/fig_7-22.png)

#### 7.5 CNN 구현하기
    ![CNN network](../images/fig_7-23.png)

#### 7.6 CNN 시각화하기
##### 7.6.1 1번째 층의 가중치 시각화하기
    ![weight](../images/fig_7-24.png)
    ![filter example](../images/fig_7-25.png)
##### 7.6.2 층 깊이에 따른 추출 정보 변화
    ![CNN layer](../images/fig_7-26.png)
#### 7.7 대표적인 CNN
##### 7.7.1 LeNet
    ![LeNet](../images/fig_7-27.png)
##### 7.7.2 AlexNet
    ![AlexNet](../images/fig_7-28.png)
#### 7.8 정리
    