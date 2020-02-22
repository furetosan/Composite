#!/usr/bin/env python
# -*- coding; utf-8 -*-

import numpy as np
# pint can also be used for keeping tabs on units
# import pint as p
# units = p.UnitRegistry()
# import sympy as sy
inv = np.linalg.inv
a = np.array
matrix = np.matrix

class Material(object):
    def __init__(self,
                 E_f = 76000.0, # MPa
                 G_f = 33000.0, # MPa
                 nu_f = 0.22,
                 E_m = 34000.0, # MPa
                 G_m = 1000.0, # MPa
                 nu_m = 0.30,
                 V_f = 0.55,
                 V_m = 0.,
                 F1t = 1080., # MPa
                 F1c = 620., # MPa
                 F2t = 39., # MPa
                 F2c = 128., # MPa
                 F6 = 89.,  # MPa
                ):
        self.E_f = E_f
        self.G_f = G_f
        self.nu_f = nu_f
        self.E_m = E_m
        self.G_m = G_m
        self.nu_m = nu_m
        self.V_f = V_f
        if V_m == 0.:
            self.V_m = 1- V_f
            self.V_v = 0.
        else:
            self.V_m = V_m
            self.V_v = 1 - V_f - V_m
            assert(self.V_v >= 0.)
        self.F1t = F1t
        self.F1c = F1c
        self.F2t = F2c
        self.F2c = F2c
        self.F6 = F6

    def calculate_engineering_constants(self):
        # Change this method to update engineering constants
        # or to fine tune the micromechanical model
        self.E_1 = self.V_f*self.E_f + self.V_m*self.E_m
        self.nu_12 = self.V_f*self.nu_f + self.V_m*self.nu_m
        self.E_2 = 1/(self.V_f/self.E_f+self.V_m/self.E_m)
        self.G_12 = 1/(self.V_f/self.G_f + self.V_m/self.G_m)

    def get_constitutive_on(self):
        S11 = 1/self.E_1
        S22 = 1/self.E_2
        S12 = -self.nu_12/self.E_1
        S66 = 1/self.G_12
        S_on = matrix([
                [S11, S12, 0.],
                [S12, S22, 0.],
                [0., 0., S66]
            ]) #, dtype=np.float64)
        nu_21 = self.nu_12*self.E_2/self.E_1
        Q11 = self.E_1/(1 - self.nu_12*nu_21)
        Q12 = self.nu_12*self.E_2/(1 - self.nu_12*nu_21)
        Q22 = self.E_1/(1 - self.nu_12*nu_21)
        Q66 = self.G_12
        Q_on = matrix([
                [Q11, Q12, 0.],
                [Q12, Q22, 0.],
                [0., 0., Q66 ]
            ]) #, dtype=np.float64)
        return S_on, Q_on
    
class Sheet(object):
    def __init__(self,
                thickness = 0.001, # m GRANDE ERRO
                theta = 0, # graus
                material = Material()):
        self.thickness = thickness
        self.theta = theta
        self.F1t = material.F1t
        self.F1c = material.F1c
        self.F2t = material.F2t
        self.F2c = material.F2c
        self.F6 = material.F6
        self.material = material

    def get_T(self):
        theta_rad = self.theta/180.*np.pi
        m,n = np.cos(theta_rad), np.sin(theta_rad)
        T = matrix([
                [ m**2, n**2, 2*m*n ],
                [ n**2, m**2, -2*m*n ],
                [ -m*n, m*n, m**2 - n**2]
            ]) # , dtype=np.float64)
        return T
    
    def get_Tinv(self):
        theta_rad = self.theta/180.*np.pi
        m,n = np.cos(theta_rad), np.sin(theta_rad)
        T = matrix([
                [ m**2, n**2, -2*m*n ],
                [ n**2, m**2, 2*m*n ],
                [ m*n, -m*n, m**2 - n**2]
            ]) # , dtype=np.float64)
        # print("T: "+str(T))
        return T

    def get_on_axis_compliance(self):
        S11 = 1/self.material.E_1
        S22 = 1/self.material.E_2
        S12 = -self.material.nu_12/self.material.E_1
        S66 = 1/self.material.G_12
        S_on = matrix([
                [S11, S12, 0.],
                [S12, S22, 0.],
                [0., 0., S66]
            ]) # , dtype=np.float64)
        return S_on

    def get_on_axis_stiffness(self):
        # self.material.calculate_engineering_constants()
        return inv(self.get_on_axis_compliance())

    def get_off_axis_compliance(self):
        T = self.get_T()
        Tinv = inv(T)
        self.calculate_constitutive_on()
        S_off = (Tinv*self.get_on_axis_compliance())*T
        return S_off

    def get_off_axis_stiffness(self):
        T = self.get_T()
        Tinv = self.get_Tinv()
        Q_off = (Tinv*self.get_on_axis_stiffness())*T
        return Q_off

class SymmetricComposite(object):
    def __init__(self,
                sheets = [Sheet()],
                thickness = 0.019): # metros
        self.pos_safety_factors = []
        self.neg_safety_factors = []
        self.sheets = sheets
        self.thickness = thickness

    def calculate_strain(self,Nmatrix,Mmatrix):
        A = self.get_A()
        D = self.get_D()
        number_of_total_sheets = 2*len(self.sheets)
        self.epsilon0 = inv(A)*Nmatrix/self.thickness
        self.kappa = inv(D)*Mmatrix/self.thickness

    # fazer a distribuicao das deformacoes "off-axis" para cada lamina
    def update_lamina_strain(self):
            number_of_total_sheets = 2*len(self.sheets)
            # Remember all sheets must have the same given thickness
            total_sheet_thickness = 2*sum( [ sheet.thickness for sheet in self.sheets ] )
            isopor_zed = (self.thickness - total_sheet_thickness)/2
            epsilon0 = self.epsilon0
            kappa = self.kappa
            for i in range(len(self.sheets)):
                delta_z = self.sheets[i].thickness
                zed = delta_z*(i+0.5)+isopor_zed
                self.sheets[i].zed = zed
                self.sheets[i].pos_epsilon_off = epsilon0+zed*kappa
                self.sheets[i].neg_epsilon_off = epsilon0-zed*kappa        

    # calcular tensoes "on-axis" para cada lamina
    def update_lamina_stress(self):
        for i in range(len(self.sheets)):
            transf_matrix = self.sheets[i].get_T()
            self.sheets[i].neg_epsilon_on = transf_matrix*self.sheets[i].neg_epsilon_off
            self.sheets[i].pos_epsilon_on = transf_matrix*self.sheets[i].pos_epsilon_off
            stiffness = self.sheets[i].get_on_axis_stiffness()
            self.sheets[i].neg_sigma_on = stiffness*self.sheets[i].neg_epsilon_on
            self.sheets[i].pos_sigma_on = stiffness*self.sheets[i].pos_epsilon_on

    # determinar e printar ambos Sf para cada lmina (neg e pos)
    def assess_safety(self):
        for sheet in self.sheets:
            f1 = 1/sheet.F1t - 1/sheet.F1c
            f2 = 1/sheet.F2t - 1/sheet.F2c
            f11 = 1/sheet.F1c/sheet.F1t
            f22 = 1/sheet.F2c/sheet.F2t
            f12 = 1/sheet.F1t/sheet.F2c
            f12 = -0.5*np.sqrt(1/sheet.F1t/sheet.F1c/sheet.F2t/sheet.F2c)
            f66 = 1/sheet.F6**2
            # lamina no lado de z negativo
            sig1 = sheet.neg_sigma_on.A[0][0]
            sig2 = sheet.neg_sigma_on.A[1][0]
            tau6 = sheet.neg_sigma_on.A[2][0]
            a = f11*sig1**2 + f22*sig2**2 + f66*tau6**2 + f12*sig1*sig2
            b = f1*sig1 + f2*sig2
            c = -1
            delta = b**2 - 4*a*c
            sheet.neg_Sf_1 = (-b+np.sqrt(delta))/2./a
            sheet.neg_Sf_2 = (-b-np.sqrt(delta))/2./a
            self.pos_safety_factors.append(sheet.neg_Sf_1)
            self.pos_safety_factors.append(sheet.neg_Sf_2)
            # lamina no lado de z positivo
            sig1 = sheet.pos_sigma_on.A[0][0]
            sig2 = sheet.pos_sigma_on.A[1][0]
            tau6 = sheet.pos_sigma_on.A[2][0]
            a = f11*sig1**2 + f22*sig2**2 + f66*tau6**2 + f12*sig1*sig2
            b = f1*sig1 + f2*sig2
            c = -1
            delta = b**2 - 4*a*c
            sheet.pos_Sf_1 = (-b+np.sqrt(delta))/2./a
            sheet.pos_Sf_2 = (-b-np.sqrt(delta))/2./a
            self.neg_safety_factors.append(sheet.pos_Sf_1)
            self.neg_safety_factors.append(sheet.pos_Sf_2)

    def populate_symmetric_positions(self):
        number_of_total_sheets = 2*len(self.sheets)
        # Remember all sheets must have the same given thickness
        total_sheet_thickness = 2*sum( [ sheet.thickness for sheet in self.sheets ] )
        zed = (self.thickness - total_sheet_thickness)/2
        for i in range(len(self.sheets)):
            delta_h = self.sheets[i].thickness
            self.sheets[i].hk = delta_h*(i+1)+zed
            self.sheets[i].hk_1 = delta_h*i+zed

##
## As contas e calculos em A, D, a e d supoe que a lista passada
## como parametro, ou posteriormente setada no objeto, esta
## organizada da seguinte forma: os elementos iniciais da list representam
## as laminas centrais dos Materiais Compositos
##
    def get_A(self):
        A = np.matrix([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
            ])
        for sheet in self.sheets:
            # double it because its symmetric
            A = A + 2*sheet.get_off_axis_stiffness()*(sheet.hk-sheet.hk_1)
        return A

    def get_D(self):
        D = np.matrix([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
            ])
        for sheet in self.sheets:
            # double it because its symmetric
            D = D + 2*sheet.get_off_axis_stiffness()*(sheet.hk**3-sheet.hk_1**3)/3
        return D

    def get_a(self):
        a = np.matrix([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
            ])
        number_of_total_sheets = 2*len(self.sheets)
        delta_h = (self.thickness/number_of_total_sheets)
        for sheet in self.sheets:
            # double it because its symmetric
            a = a + 2*sheet.get_off_axis_stiffness()*(sheet.hk-sheet.hk_1)
        return a/delta_h

    def get_d(self):
        d = np.matrix([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
            ])
        number_of_total_sheets = 2*len(self.sheets)
        delta_h = (self.thickness/number_of_total_sheets)
        for sheet in self.sheets:
            # double it because its symmetric
            d = d + 2*sheet.get_off_axis_stiffness()*(sheet.hk**3-sheet.hk_1**3)/3
        return 12*d/delta_h**3