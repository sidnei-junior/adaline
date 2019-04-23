#Implementação Adaline

import random

class Adaline:

    def __init__(self, amostras, saidas, taxa_aprendizado = 0.1, taxa_precisao = 0.1, epocas = 1000, limiar = -1):
        self.amostras = amostras
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.taxa_precisao = taxa_precisao
        self.epocas = epocas
        self.limiar = limiar
        self.n_amostras = len(amostras)
        self.n_atributos = len (amostras[0])
        self.pesos = []
        
    # Training Function
    def train(self):

        for amostra in self.amostras:
            amostra.insert(0, -1)

        for i in range(self.n_atributos+1):
            self.pesos.append(random.random())

        self.pesos.insert(0, self.limiar)
        n_epocas = 0 # contador de épocas

        while n_epocas <= 1000:
            mse_p = self.mse() # previous mse
            potencial_ativacao = []
            for i in range(self.n_amostras):
                u = 0
                for j in range(self.n_atributos+1):
                    u += self.pesos[j]*self.amostras[i][j]
                #y = self.signal(u)
                potencial_ativacao.append(u)

                for k in range(self.n_atributos+1):
                    self.pesos[k] = self.pesos[k] + self.taxa_aprendizado*(self.saidas[i]-u)*self.amostras[i][k]
            mse_c = self.mse() # current mse
            n_epocas += 1
            if abs(mse_c-mse_p)<=self.taxa_precisao:
                break
            
        print(n_epocas)
        print(abs(mse_c-mse_p))

    def teste(self, amostra):
        amostra.insert(0, -1)
        u = 0
        for i in range(self.n_atributos + 1):
            u += self.pesos[i] * amostra[i]
        y = self.signal(u)
        print ('Classe: %d' %y)

    # Mean Square Error Function
    def mse(self):
        eqm = 0
        for i in range(self.n_amostras):
            u = 0
            for j in range(self.n_atributos+1):
                u += self.pesos[j]*self.amostras[i][j]
            eqm += (self.saidas[i]-u)**2
        eqm /= self.n_amostras
        return eqm
        



    # Signal Function
    def signal(self, u):
        if u > 0:
            return 1
        elif u == 0:
            return 0
        return -1


# no problema em questão

aprendizado = 0.0025
precisao = 0.000001
amostras  = [[0.4329,-1.3719, 0.7022, -0.8535],
            [0.3024, 0.2286, 0.8630, 2.7909],
            [0.1349, -0.6445, 1.0530, 0.5687],
            [0.3374,-1.7163,0.3670,-0.6283],
            [1.1434, -0.0485, 0.6637, 1.2606],
            [1.3749, -0.5071, 0.4464, 1.3009],
            [0.7221, -0.7587, 0.7681, -0.5592],
            [0.4403, -0.8072, 0.5154, -0.3129],
            [-0.5231, 0.3548, 0.2538, 1.5776],
            [0.3255, -2, 0.7112, -1.1209],
            [0.5824, 1.3915, -0.2291, 4.1735],
            [0.134, 0.6081, 0.445, 3.223],
            [0.1480, -0.2988, 0.4778, 0.8643],
            [0.7359, 0.1869, -0.0872, 2.3584],
            [0.7115, -1.1469, 0.3394, 0.9573],
            [0.8251, -1.284, 0.8452, 1.2382],
            [0.1569, 0.3712, 0.8825, 1.7633],
            [0.0033, 0.6835, 0.5389, 2.8249],
            [0.4243, 0.8313, 0.2634, 3.5855],
            [1.049, 0.1326, 0.9138, 1.9792],
            [1.4276, 0.5331, -0.0145, 3.7286],
            [0.5971, 1.4865, 0.2904, 4.6069],
            [0.8475, 2.1479, 0.3179, 5.8235],
            [1.3967, -0.4171, 0.6443, 1.3927],
            [0.0044, 1.5378, 0.6099, 4.7755],
            [0.2201, -0.5668, 0.0515, 0.7829],
            [0.63, -1.248, 0.8591, 0.8093],
            [-0.2479, 0.896, 0.0547, 1.7381],
            [-0.3088,-0.0929, 0.8659, 1.5483],
            [-0.518,1.4974, 0.5453, 2.3993],
            [0.6833, 0.8266, 0.0829, 2.8864],
            [0.4353, -1.4066, 0.4207, -0.4879],
            [-0.1069, -3.2329, 0.1856, -2.4572],
            [0.4662, 0.6261, 0.7304, 3.437],
            [0.8298, -1.4089, 0.3119, 1.3235]]

saidas = [1, -1, -1, -1, 1, 1, 1, 1, -1, 1,
          -1, -1, 1,  1,  -1,  -1,  1,  -1,  -1,  1,
          1, -1, -1,  1,  -1,  1,  -1,  1,  -1,  1,
           1,  1,  -1,  -1,  -1]

rede = Adaline(amostras, saidas, aprendizado, precisao)
rede.train()
rede.teste([0.4353, -1.4066, 0.4207, -0.4879])


        