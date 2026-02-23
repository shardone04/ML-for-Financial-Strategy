from Keras.models import Sequential
from Keras.layers import Dense
import numpy as np

data = np.random.random((1000,10))
Y = np.random.randint(2, size = (1000, 1))
model = Sequential

# 모델 생성 - 신경망 구조

# 모델에는 10개의 변수(input_dim = 10인수)가 있다.
# 첫 번째 은닉층은 노드가 32개 있고 relu 활성화 함수를 사용한다.
# 두 번째 은닉층은 노드가 32개 있고 relu 활성화 함수를 사용한다.
# 출력층은 노드가 하나 있고 sigmoid 활성화 함수를 사용한다.

model = Sequential
model.add(Dense(32, input_dim = 10, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일 : 테아노나 텐서플로 패키지에 있는 효율적인 수치 계산 라이브러리를 이용한다는 의미
model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , \
              metrics = ['accuracy'])

# 모델 적합화
model.fit(data, Y, nb_epoch = 10, batch_size = 32)

# 모델 평가
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

