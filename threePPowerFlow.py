import autograd.numpy as np

def f_to_m(feet):
  miles=feet*.00018939394
  return miles

def Pflow(Vi,Vj,Y):
  G = np.real(Y)
  B = np.imag(Y)
  magi = np.abs(Vi)
  magj = np.abs(Vj)
  anglei = np.angle(Vi)
  anglej = np.angle(Vj)

  Pi = magi.T.dot(G*np.cos(anglei.T-anglei)+B*np.sin(anglei.T-anglei))
  Pj = magj.T.dot(G*np.cos(anglei.T-anglej)+B*np.sin(anglei.T-anglej))

  return magi*(Pi.T-Pj.T)

def Qflow(Vi,Vj,Y):
  G = np.real(Y)
  B = np.imag(Y)
  magi = np.abs(Vi)
  magj = np.abs(Vj)
  anglei = np.angle(Vi)
  anglej = np.angle(Vj)

  Qi = magi.T.dot(G*np.sin(anglei.T-anglei)-B*np.cos(anglei.T-anglei))
  Qj = magj.T.dot(G*np.sin(anglei.T-anglej)-B*np.cos(anglei.T-anglej))

  return magi*(+Qi.T-Qj.T)

def Ptrflow(magi,anglei,magj,anglej,z):
  y = 1/z
  g = np.real(y)
  b = np.imag(y)
  # magi = np.abs(Vi)
  # magj = np.abs(Vj)
  # anglei = np.angle(Vi)
  # anglej = np.angle(Vj)

  X = magi**2-magi*magj*np.cos(anglei-anglej)
  Y = - magi*magj*np.sin(anglei-anglej)
  P = g*X+b*Y

  return P

def Qtrflow(magi,anglei,magj,anglej,z):
  y = 1/z
  g = np.real(y)
  b = np.imag(y)
  # magi = np.abs(Vi)
  # magj = np.abs(Vj)
  # anglei = np.angle(Vi)
  # anglej = np.angle(Vj)

  X = magi**2-magi*magj*np.cos(anglei-anglej)
  Y = - magi*magj*np.sin(anglei-anglej)
  Q = -b*X+g*Y

  return Q

if __name__ == '__main__':

  A = np.sqrt(2/3)*np.array([
                              [1,0,1/np.sqrt(2)],
                              [-1/2,np.sqrt(3)/2,1/np.sqrt(2)],
                              [-1/2,-np.sqrt(3)/2,1/np.sqrt(2)]
                            ])
  zy = np.array([
                  [0.4576+1.078j,0.1559+0.5017j,0.1535+0.3849j],
                  [0.1559+0.5017j,0.4666+1.0482j,0.158+0.4236j],
                  [0.1535+0.3849j,0.158+0.4236j,0.4615+1.0651j]
                ])
  zd = np.array([
                  [0.4013+1.4133j,0.0953+0.8515j,0.0953+0.7266j],
                  [0.0953+0.8515j,0.4013+1.4133j,0.0953+0.7802j],
                  [0.0953+0.7266j,0.0953+0.7802j,0.4013+1.4133j]
                ])
  tr = (1+6j)*np.identity(3)/100

  V1 = 12470/np.sqrt(3)*np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]]))
  # Power Flow from Bus 1 to Bus 2
  V2 = np.array([[7164,7110,7082]]).T*np.exp(1j*np.deg2rad([[-0.1,-120.2,119.3]])).T
  Z12 = zy*f_to_m(2000)
  Y12 = np.linalg.inv(Z12)
  
  P = Pflow(V1,V2,Y12)
  Q = Qflow(V1,V2,Y12)
  print('--- P12 ---')
  print(P)
  print('--- Q12 ---')
  print(Q)

  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T
  V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T

  Z34 = zy*f_to_m(2500)
  Y34 = np.linalg.inv(Z34)
  # Y34 = 1/Z34
  P = Pflow(V4,V3,Y34)
  Q = Qflow(V4,V3,Y34)
  print('--- P43 ---')
  print(P)
  print('--- Q43 ---')
  print(Q)
  print(np.cos(np.angle(P+1j*Q)))
  P = Pflow(V3,V4,Y34)
  Q = Qflow(V3,V4,Y34)
  print('--- P34 ---')
  print(P)
  print('--- Q34 ---')
  print(Q)
  print(np.cos(np.angle(P+1j*Q)))
  s = np.array([[1275000/0.85,1800000/0.9,2375000/0.95]]).T*np.exp(1j*np.arccos([[0.85,0.9,0.95]])).T
  print(s)
  # P34_1 =  V4[0,0]*(V4[1,0]*)