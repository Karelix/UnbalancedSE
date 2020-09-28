import autograd.numpy as np

def Pflow(Vi,deltai,Vj,deltaj,Y,d):
  G = np.real(Y)
  B = np.imag(Y)

  V = np.concatenate((Vi,Vj),axis=0)
  delta = np.concatenate((deltai,deltaj),axis=0)

  p = []
  off = 0 if d == 'ij' else 3
  for i in range(3):
    temp = 0
    for k in range(6):
      temp = temp + V[k,0]*(G[i+off,k]*np.cos(delta[i+off,0]-delta[k,0])+B[i+off,k]*np.sin(delta[i+off,0]-delta[k,0]))
    p = p + [[V[i+off,0]*temp]]

  return np.array(p)

def Qflow(Vi,deltai,Vj,deltaj,Y,d):
  G = np.real(Y)
  B = np.imag(Y)

  V = np.concatenate((Vi,Vj),axis=0)
  delta = np.concatenate((deltai,deltaj),axis=0)

  q = []
  off = 0 if d == 'ij' else 3
  for i in range(3):
    temp = 0
    for k in range(6):
      temp = temp + V[k,0]*(G[i+off,k]*np.sin(delta[i+off,0]-delta[k,0])-B[i+off,k]*np.cos(delta[i+off,0]-delta[k,0]))
    q = q + [[V[i+off,0]*temp]]

  return np.array(q)