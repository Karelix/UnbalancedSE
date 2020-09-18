import autograd.numpy as np
from autograd import jacobian
import numpy as nu
# import numpy as np
from threePPowerFlow import f_to_m,Ptrflow,Qtrflow

def Pflow(Vi,deltai,Vj,deltaj,Y):
  G = np.real(Y)
  B = np.imag(Y)

  p = []
  for k in range(3):
    temp = 0
    for z in range(3):
      temp = temp + Vi[z,0]*(G[k,z]*np.cos(deltai[k,0]-deltai[z,0])+B[k,z]*np.sin(deltai[k,0]-deltai[z,0]))
      temp = temp - Vj[z,0]*(G[k,z]*np.cos(deltai[k,0]-deltaj[z,0])+B[k,z]*np.sin(deltai[k,0]-deltaj[z,0]))
    # print(Vi[k,0])
    p = p + [[Vi[k,0]*temp]]

  # Pi = magi.T.dot(G*np.cos(anglei.T-anglei)+B*np.sin(anglei.T-anglei))
  # Pj = magj.T.dot(G*np.cos(anglei.T-anglej)+B*np.sin(anglei.T-anglej))

  return np.array(p)

def Qflow(Vi,deltai,Vj,deltaj,Y):
  G = np.real(Y)
  B = np.imag(Y)
  # magi = np.abs(Vi)
  # magj = np.abs(Vj)
  # anglei = np.angle(Vi)
  # anglej = np.angle(Vj)

  q = []
  for k in range(3):
    temp = 0
    for z in range(3):
      temp = temp + Vi[z,0]*(G[k,z]*np.sin(deltai[k,0]-deltai[z,0])-B[k,z]*np.cos(deltai[k,0]-deltai[z,0]))
      temp = temp - Vj[z,0]*(G[k,z]*np.sin(deltai[k,0]-deltaj[z,0])-B[k,z]*np.cos(deltai[k,0]-deltaj[z,0]))
    q = q + [[Vi[k,0]*temp]]

  # Qi = magi.T.dot(G*np.sin(anglei.T-anglei)-B*np.cos(anglei.T-anglei))
  # Qj = magj.T.dot(G*np.sin(anglei.T-anglej)-B*np.cos(anglei.T-anglej))

  # return magi*(+Qi.T-Qj.T)
  return np.array(q)

def equations(x):
  s_base = 6000000
  zy = np.array([
                  [0.4576+1.078j,0.1559+0.5017j,0.1535+0.3849j],
                  [0.1559+0.5017j,0.4666+1.0482j,0.158+0.4236j],
                  [0.1535+0.3849j,0.158+0.4236j,0.4615+1.0651j]
                ])
  v_base = 12470
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)
  z12 = zy*f_to_m(2000)/z_base

  v_base = 4160
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)
  z34 = zy*f_to_m(2500)/z_base

  y12 = np.linalg.inv(z12)
  y34 = np.linalg.inv(z34)

  eqs = np.empty((0,1))
  x = np.array([list(x)]).T
  # --------------------------------------------------------------------
  # Flow 1-2
  # vi = x[0:3,:]*np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]]))
  # vj = x[3:6,:]*np.exp(1j*x[6:9,:])
  # p = Pflow(vi,vj,y12)
  p = Pflow(x[0:3,:],np.array([[0],[-2*np.pi/3],[2*np.pi/3]]),x[3:6,:],x[6:9,:],y12)
  q = Qflow(x[0:3,:],np.array([[0],[-2*np.pi/3],[2*np.pi/3]]),x[3:6,:],x[6:9,:],y12)
  eqs = np.concatenate((eqs,p,q),axis=0)
  # --------------------------------------------------------------------
  # Transformer Flow 2-3
  vi = x[3:6,:]*np.exp(1j*x[6:9,:])
  vj = x[9:12,:]*np.exp(1j*x[12:15,:])
  p = Ptrflow(x[3:6,:],x[6:9,:],x[9:12,:],x[12:15,:],(1+6j)/100)
  q = Qtrflow(x[3:6,:],x[6:9,:],x[9:12,:],x[12:15,:],(1+6j)/100)
  eqs = np.concatenate((eqs,p,q),axis=0)
  # --------------------------------------------------------------------
  # Flow 3-4
  vi = x[9:12,:]*np.exp(1j*x[12:15,:])
  vj = x[15:18,:]*np.exp(1j*x[18:21,:])
  p = Pflow(x[9:12,:],x[12:15,:],x[15:18,:],x[18:21,:],y34)
  q = Qflow(x[9:12,:],x[12:15,:],x[15:18,:],x[18:21,:],y34)
  eqs = np.concatenate((eqs,p,q),axis=0)
  # --------------------------------------------------------------------
  # Injection 4
  v4 = x[15:18,:]*np.exp(1j*x[18:21,:])
  v3 = x[9:12,:]*np.exp(1j*x[12:15,:])
  G = np.real(y34)
  B = np.imag(y34)
  # p = x[15:18,:]*(x[15:18,:].T.dot(G*np.cos(x[18:21,:].T-x[18:21,:])+B*np.sin(x[18:21,:].T-x[18:21,:]))+
  #                 x[9:12,:].T.dot(G*np.cos(x[18:21,:].T-x[12:15,:])+B*np.sin(x[18:21,:].T-x[12:15,:]))).T
  # q = x[15:18,:]*(x[15:18,:].T.dot(G*np.sin(x[18:21,:].T-x[18:21,:])-B*np.cos(x[18:21,:].T-x[18:21,:]))+
  #                 x[9:12,:].T.dot(G*np.sin(x[18:21,:].T-x[12:15,:])-B*np.cos(x[18:21,:].T-x[12:15,:]))).T
  p = -Pflow(x[15:18,:],x[18:21,:],x[9:12,:],x[12:15,:],y34)
  q = -Qflow(x[15:18,:],x[18:21,:],x[9:12,:],x[12:15,:],y34)
  eqs = np.concatenate((eqs,p,q),axis=0)
  eqs = np.concatenate((eqs,x[0:3,:]),axis=0)
  # --------------------------------------------------------------------

  return eqs

def printA(a):
  for row in a:
    for col in row:
      print("{:8.3f}".format(col), end=" ")
    print("")

if __name__ == '__main__':
  s_base = 6000000
  zy = np.array([
                  [0.4576+1.078j,0.1559+0.5017j,0.1535+0.3849j],
                  [0.1559+0.5017j,0.4666+1.0482j,0.158+0.4236j],
                  [0.1535+0.3849j,0.158+0.4236j,0.4615+1.0651j]
                ])
  tr = (1+6j)/100

  v_base = 12470
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)

  V1 = 12470/np.sqrt(3)*np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]]))/(v_base/np.sqrt(3))
  # Power Flow from Bus 1 to Bus 2
  V2 = np.array([[7164,7110,7082]]).T*np.exp(1j*np.deg2rad([[-0.1,-120.2,119.3]])).T/(v_base/np.sqrt(3))
  Z12 = zy*f_to_m(2000)/z_base
  Y12 = np.linalg.inv(Z12)
  
  P = Pflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),Y12)
  Q = Qflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),Y12)
  
  meas = np.empty((0,1))
  meas = np.concatenate((meas,P,Q),axis=0)

  v_base = 4160
  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T/(v_base)
  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T/(v_base/np.sqrt(3))
  P = Ptrflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),tr)
  Q = Qtrflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),tr)
  meas = np.concatenate((meas,P,Q),axis=0)

  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)

  V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T/(v_base/np.sqrt(3))
  # V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T/v_base
  # Power Flow from Bus 3 to Bus 4
  Z34 = zy*f_to_m(2500)/z_base
  # Z34 = zy*f_to_m(2500)
  Y34 = np.linalg.inv(Z34)
  P = Pflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),Y34)
  Q = Qflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),Y34)
  meas = np.concatenate((meas,P,Q),axis=0)

  # Injection 4
  mag4 = np.abs(V4)
  mag3 = np.abs(V3)
  angle4 = np.angle(V4)
  angle3 = np.angle(V3)
  G = np.real(Y34)
  B = np.imag(Y34)
  # p = mag4*(mag4.T.dot(G*np.cos(angle4.T-angle4)+B*np.sin(angle4.T-angle4))+
  #                 mag3.T.dot(G*np.cos(angle4.T-angle3)+B*np.sin(angle4.T-angle3))).T
  # q = mag4*(mag4.T.dot(G*np.sin(angle4.T-angle4)-B*np.cos(angle4.T-angle4))+
  #                 mag3.T.dot(G*np.sin(angle4.T-angle3)-B*np.cos(angle4.T-angle3))).T
  p = np.array([[1275000,1800000,2375000]]).T/(s_base/3)
  c = np.array([[0.85,0.9,0.95]]).T
  q = p/c*np.sin(np.arccos(c))
  meas = np.concatenate((meas,p,q),axis=0)
  meas = np.concatenate((meas,np.abs(V1)),axis=0)
  # print(meas)

  P = Pflow(np.abs(V4),np.angle(V4),np.abs(V3),np.angle(V3),Y34)
  Q = Qflow(np.abs(V4),np.angle(V4),np.abs(V3),np.angle(V3),Y34)
  print(P)
  print(Q)
  print(P*s_base)
  # P = Pflow(np.abs(V4*v_base/np.sqrt(3)),np.angle(V4*v_base/np.sqrt(3)),np.abs(V3*v_base/np.sqrt(3)),np.angle(V3*v_base/np.sqrt(3)),Y34/z_base)
  # Q = Qflow(np.abs(V4*v_base/np.sqrt(3)),np.angle(V4*v_base/np.sqrt(3)),np.abs(V3*v_base/np.sqrt(3)),np.angle(V3*v_base/np.sqrt(3)),Y34/z_base)
  # print(P)
  # print(Q)
  # input()
  # P21 = Pflow(np.abs(V2),np.angle(V2),np.abs(V1),np.angle(V1),Y12)
  Q21 = Qflow(np.abs(V2),np.angle(V2),np.abs(V1),np.angle(V1),Y12)
  Q23 = Qtrflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),tr)

  
  # print(Q21+Q23)
  # input()
  # -------------------------------------------------------------------------------------------
  W = np.identity(meas.shape[0])*100
  for i in range(18,27): 
    W[i,i] = 1000000000

  x = np.array([1,1,1,
       1,1,1,0,-2*np.pi/3,2*np.pi/3,
       1,1,1,0,-2*np.pi/3,2*np.pi/3,
       1,1,1,0,-2*np.pi/3,2*np.pi/3])

  # x = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),
  #      1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0,-2*np.pi/3,2*np.pi/3,
  #      1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0,-2*np.pi/3,2*np.pi/3,
  #      1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0,-2*np.pi/3,2*np.pi/3])

  jac = jacobian(equations)
  err = []
  i = 0
  e = 10
  while np.abs(e) > 0.00001 and i < 100:
    print(i)
    H = jac(x).reshape(-1,len(list(x)))
    # printA(H)
    # input()
    G = H.T.dot(W).dot(H)
    # print(np.linalg.det(H.T.dot(H)))
    # input()
    b = H.T.dot(W).dot(meas-equations(x))
    dx = nu.linalg.solve(G,b)
    # print(dx)
    e = np.sum(dx)/len(dx)
    print(e)
    err.append(e)
    i = i + 1
    if np.abs(e) > 0.00001:
      x = x + dx.flatten()
    
  # print(x)
  for i in range(3):
    v_base = 12470 if i == 0 else 4160
    print('----- V'+str(i+2)+' -----')
    for j,ph in zip(range(3),['a: ','b: ','c: ']):
      print(ph,end='')
      v = round(x[3+6*i+j]*v_base/np.sqrt(3),3)
      d = round(np.degrees(x[6+6*i+j]),3)
      print(str(v)+'/'+str(d))
