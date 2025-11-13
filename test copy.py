from backward import *

x = np.array([1])
y = np.array([20])
import time
L1 = Layer(1, 1)
# L2 = Layer(2, 1)
opti = Adam(L1.parameters() , 0.0001)
for i in range(1000):
    # time.sleep(0.5)
    # for idx in range(x.shape[0]):
        print("------------------")
        print(x)
        a2 = L1(x)
        # a2 = L2(ActivationFunction.RelU(a1))
        print(f"a2: {a2}")
        print(f"y: {y}")
        loss = y - a2
        print(f"lossList{loss}")
        l=Value(0)
        for i in range(len(loss)):
            l+=loss[i]**2
        # print(f"loss:{l}")
        opti.zero_grad()
        l.backward()
        opti.step()
        print(opti.getParameters()[0].grad)
        print(f"loss:{l}")







