#
# Name:         TP1.py
# Purpose:      Code du TP1 de génie mathématique (2ème semestre Aéro 2)
#
# Authors:      MARANDOLA Lorris    COULOT Alexandre    ALLIO Paul
#
# Last edited:  16/03/2021


### IMPORTS ###


import numpy as np
import time
import matplotlib.pyplot as pl


### Gauss ###


def ReductionGauss(Aaug):
    taille_matrice = np.shape(Aaug)
    taille_matrice_ligne = taille_matrice[0]
    taille_matrice_colonne = taille_matrice[1]

    for k in range(0, taille_matrice_ligne - 1):

        for j in range(k, taille_matrice_colonne - 2):
            g = Aaug[j + 1, k] / Aaug[k, k]
            Aaug[j + 1] = Aaug[j + 1] - g * Aaug[k]
            g = Aaug[j, k] / Aaug[k, k]

    return Aaug


def ResolutionSystTriSup(Taug):
    taille_matrice = np.shape(Taug)
    taille_matrice_ligne = taille_matrice[0]
    taille_matrice_colonne = taille_matrice[1]

    X = np.zeros((taille_matrice_ligne, 1))

    for i in range(taille_matrice_ligne - 1, -1, -1):

        for k in range(i + 1, taille_matrice_colonne - 1):
            Taug[i, taille_matrice_colonne - 1] = Taug[i, taille_matrice_colonne - 1] - Taug[i, k]

        inconnue = Taug[i, taille_matrice_colonne - 1] / Taug[i, i]

        for k in range(0, taille_matrice_ligne):
            Taug[k, i] = Taug[k, i] * inconnue

        X[i] = inconnue

    return X


def Gauss(A, B):
    Aaug = np.c_[A, B]

    T = ReductionGauss(Aaug)
    X = ResolutionSystTriSup(T)

    return X


### LU ###


def DecompositionLU(A):
    A_ = np.copy(A)

    taille_matrice = np.shape(A_)
    taille_matrice_ligne = taille_matrice[0]
    taille_matrice_colonne = taille_matrice[1]
    L = np.eye(taille_matrice_ligne)

    for k in range(0, taille_matrice_ligne - 1):

        for i in range(k + 1, taille_matrice_ligne):
            g = A_[i, k] / A_[k, k]

            for j in range(k, taille_matrice_colonne):
                A_[i, j] = A_[i, j] - g * A_[k, j]
                L[i, k] = g
    return L, A_


def ResolutionLU(L, U, B):
    taille_matrice = np.shape(L)[0]
    LB = np.c_[L, B]
    Y = np.zeros((taille_matrice, 1))

    for ligne in range(taille_matrice):

        Y[ligne] = LB[ligne, taille_matrice]

        for colonne in range(ligne):
            Y[ligne] = Y[ligne] - LB[ligne, colonne] * Y[colonne]

    UY = np.c_[U, Y]
    X = ResolutionSystTriSup(UY)

    return X


def LU(A, B):
    L = DecompositionLU(A)[0]
    U = DecompositionLU(A)[1]
    X = ResolutionLU(L, U, B)

    return X


### Partiel ###


def GaussChoixPivotPartiel(A, B):
    Aaug = np.c_[A, B]

    taille_matrice = np.shape(Aaug)
    taille_matrice_ligne = taille_matrice[0]
    taille_matrice_colonne = taille_matrice[1]

    for k in range(0, taille_matrice_ligne - 1):
        pivot_max = Aaug[k, k]
        indice = k

        for i in range(k + 1, taille_matrice_ligne):

            if abs(Aaug[i, k]) > abs(pivot_max):
                pivot_max = Aaug[i, k]
                indice = i

                transfert = np.copy(Aaug[k])
                Aaug[k] = Aaug[i]
                Aaug[i] = transfert

        for j in range(k, taille_matrice_colonne - 2):
            g = Aaug[j + 1, k] / Aaug[k, k]
            Aaug[j + 1] = Aaug[j + 1] - g * Aaug[k]

    X = ResolutionSystTriSup(Aaug)

    return X


### Total ###


def GaussChoixPivotTotal(A, B):
    Aaug = np.c_[A, B]

    taille_matrice = np.shape(Aaug)
    taille_matrice_ligne = taille_matrice[0]
    taille_matrice_colonne = taille_matrice[1]

    index_avant = []
    index_apres = []

    for k in range(0, taille_matrice_ligne):
        pivot_max = Aaug[k, k]
        indice = k

        for i in range(k, taille_matrice_ligne):
            if abs(Aaug[k, i]) > abs(pivot_max):
                pivot_max = Aaug[k, i]
                indice = i

                transfert = np.copy(Aaug[:, k])
                Aaug[:, k] = Aaug[:, i]
                Aaug[:, i] = transfert
                index_avant.append(i)
                index_apres.append(k)

        for j in range(k, taille_matrice_colonne - 2):
            g = Aaug[j + 1, k] / Aaug[k, k]
            Aaug[j + 1] = Aaug[j + 1] - g * Aaug[k]

    X = ResolutionSystTriSup(Aaug)

    for l in range(len(index_avant) - 1, -1, -1):
        transfert = np.copy(X[index_apres[l]])
        X[index_apres[l]] = X[index_avant[l]]
        X[index_avant[l]] = transfert

    return X


### MAIN ###


""" 
A = np.array([[3., 2, -1, 4],
             [-3, -4, 4, -2],
             [6, 2, 2, 7],
             [9, 4, 2, 18,]])

B = np.array([4 ,-5, -2, 13])

A = np.array([[1., 1, 0],
             [-1, -1, 1],
             [0, 1, 1,],])

B = np.array([0 ,1 , 2])
"""


### ERREUR ###


def Erreur_matrice(A, B, X):
    produit = np.dot(A, X)

    for ligne in range(len(produit)):
        produit[ligne] = produit[ligne] - B[ligne]

    erreur = np.linalg.norm(produit)

    return erreur


### MATRICES ###


Taille_matrice = []

GaussTps = []
LUTps = []
PartielTps = []
TotalTps = []
LinalgTps = []

GaussErr = []
LUErr = []
PartielErr = []
TotalErr = []
LinalgErr = []

TempsTotal = time.time()

for matrices in range(200, 1050, 50):
    A = np.random.rand(matrices, matrices)
    B = np.random.rand(matrices)

    TG1 = time.time()
    Xgauss = Gauss(A, B)
    TG2 = time.time()
    TG = TG2 - TG1
    GaussTps.append(TG)
    ErreurGauss = Erreur_matrice(A, B, Xgauss)
    GaussErr.append(ErreurGauss)

    TLU1 = time.time()
    Xlu = LU(A, B)
    TLU2 = time.time()
    TLU = TLU2 - TLU1
    LUTps.append(TLU)
    ErreurLU = Erreur_matrice(A, B, Xlu)
    LUErr.append(ErreurLU)

    TP1 = time.time()
    Xpartiel = GaussChoixPivotPartiel(A, B)
    TP2 = time.time()
    TP = TP2 - TP1
    PartielTps.append(TP)
    ErreurPartiel = Erreur_matrice(A, B, Xpartiel)
    PartielErr.append(ErreurPartiel)

    TT1 = time.time()
    Xtotal = GaussChoixPivotTotal(A, B)
    TT2 = time.time()
    TT = TT2 - TT1
    TotalTps.append(TT)
    ErreurTotal = Erreur_matrice(A, B, Xtotal)
    TotalErr.append(ErreurTotal)

    TL1 = time.time()
    Xlinalg = np.linalg.solve(A, B)
    TL2 = time.time()
    TL = TL2 - TL1
    LinalgTps.append(TL)
    ErreurLinalg = Erreur_matrice(A, B, Xlinalg)
    LinalgErr.append(ErreurLinalg)

    Taille_matrice.append(matrices)

TempsTotal = time.time() - TempsTotal
print("Temps total :", TempsTotal, "s")


### GRAPH ###


pl.plot(Taille_matrice, GaussTps, label="Gauss")
pl.plot(Taille_matrice, LUTps, label="LU")
pl.plot(Taille_matrice, PartielTps, label="Partiel")
pl.plot(Taille_matrice, TotalTps, label="Total")
pl.plot(Taille_matrice, LinalgTps, label="Linalg")
pl.yscale("log")
pl.xlabel("Taille de la matrice n*n")
pl.ylabel("Temps (s)")
pl.title("Temps de résolutuion en fonction de la taille d'une matrice")
pl.legend()
pl.show()
pl.plot(Taille_matrice, GaussErr, label="Gauss")
pl.plot(Taille_matrice, LUErr, label="LU")
pl.plot(Taille_matrice, PartielErr, label="Partiel")
pl.plot(Taille_matrice, TotalErr, label="Total")
pl.plot(Taille_matrice, LinalgErr, label="Linalg")
pl.yscale("log")
pl.xlabel("Taille de la matrice n*n")
pl.ylabel("Erreur")
pl.title("Erreur en fonction de la taille d'une matrice")
pl.legend()
pl.show()
