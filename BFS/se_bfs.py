import autograd.numpy as np
from autograd import jacobian
import numpy as nu
from flows import *
from network import *

def printA(a):
  for row in a:
    for col in row:
      print("{:8.3f}".format(col), end=" ")
    print("")

def init_states(buses,x):
  t = x
  descs = []
  for bus in buses:
    if not bus.slack:
      t = np.concatenate((t,bus.v_states,bus.delta_states),axis=0)
    descs = descs + bus.desc
  if len(descs) == 0:
    return t
  else:
    return init_states(descs,t)

def back_equations(x,ref,y,flag):
  eqs = np.empty((0,1))
  Vi,deltai = ref[0:3], ref[3:6]
  Vj,deltaj = x[0:3], x[3:6]
  if flag:
    p = Pflow(Vi,deltai,Vj,deltaj,y,'ij')
    q = Qflow(Vi,deltai,Vj,deltaj,y,'ij')
    eqs = np.concatenate((eqs,p,q),axis=0)
  p = Pflow(Vi,deltai,Vj,deltaj,y,'ji')
  q = Qflow(Vi,deltai,Vj,deltaj,y,'ji')
  eqs = np.concatenate((eqs,p,q),axis=0)
  return eqs

def bse(net,anc,curr,jac):
  s,sigmaP,sigmaQ = np.zeros((3,1)), np.ones((3,1))*0.000000001**2, np.ones((3,1))*0.000000001**2
  for d in curr.desc:
    ts,tsigmaP,tsigmaQ = bse(net,curr,d,jac)
    s = s + ts
    sigmaP = sigmaP + tsigmaP
    sigmaQ = sigmaQ + tsigmaQ

  if anc is None:
    return None,None,None
  else:
    line = net.lines[(anc,curr)]
    meas = np.empty((0,1))
    sigmas = np.empty((0,1))
    flag = False
    if 'ij' in line.meas:
      meas = np.concatenate((meas,line.meas['ij']['p'].m,line.meas['ij']['q'].m),axis=0)
      sigmas = np.concatenate((sigmas,line.meas['ij']['p'].sigma**2,line.meas['ij']['q'].sigma**2),axis=0)
      flag = True
    p_inj = np.real(s) + curr.p_inj.m
    q_inj = np.imag(s) + curr.q_inj.m
    meas = np.concatenate((meas,p_inj,q_inj),axis=0)
    sigmas = np.concatenate((sigmas,sigmaP,sigmaQ),axis=0)
    W = sigmas*np.identity(sigmas.shape[0])
    x = np.concatenate((curr.v_states,curr.delta_states),axis=0)
    ref = np.concatenate((anc.v_states,anc.delta_states),axis=0)
    y = line.y_mtx
    err = []
    i = 0
    e = 10
    while np.abs(e) > 0.00001 and i < 100:
      H = jac(x,ref,y,flag).reshape(-1,len(list(x)))
      G = H.T.dot(W).dot(H)
      b = H.T.dot(W).dot(meas-back_equations(x,ref,y,flag))
      dx = nu.linalg.solve(G,b)
      e = np.sum(dx)/len(dx)
      # print(e)
      err.append(e)
      i = i + 1
      x = x + dx

    curr.v_states = x[0:3]
    curr.delta_states = x[3:6]
    p = Pflow(anc.v_states,anc.delta_states,curr.v_states,curr.delta_states,y,'ij')
    q = Qflow(anc.v_states,anc.delta_states,curr.v_states,curr.delta_states,y,'ij')
    new_sigmaP,new_sigmaQ = sigmaP,sigmaQ 
    if flag:
      measP = line.meas['ij']['p'].m
      measQ = line.meas['ij']['q'].m
      new_sigmaP = np.min((line.meas['ij']['p'].sigma**2,(p-measP)**2),axis=0)
      new_sigmaP = np.min((line.meas['ij']['q'].sigma**2,(q-measQ)**2),axis=0)
    line.p_eq = Measurement(t='PFij', m=p, sigma=np.sqrt(new_sigmaP))
    line.q_eq = Measurement(t='QFij', m=q, sigma=np.sqrt(new_sigmaQ))
    return p+1j*q, new_sigmaP, new_sigmaQ

def print_states(buses):
  descs = []
  for bus in buses:
    if not bus.slack:
      print(bus.name)
      for i in range(3):
        print(str(bus.v_states[i,0]*bus.v_base)+'/'+str(np.degrees(bus.delta_states[i,0])))
    descs = descs + bus.desc
  if len(descs) == 0:
    return 
  else:
   print_states(descs)

def backward(net):
  jac = jacobian(back_equations)
  _,_,_ = bse(net,None,net.buses[1],jac)
  # print_states([net.buses[1]])

def fwd_equations(x,ref,y):
  Vi,deltai = ref[0:3], ref[3:6]
  Vj,deltaj = x[0:3], x[3:6]
  p = Pflow(Vi,deltai,Vj,deltaj,y,'ij')
  q = Qflow(Vi,deltai,Vj,deltaj,y,'ij')
  eqs = np.concatenate((p,q),axis=0)
  return eqs

def fse(net,anc,curr,jac):
  if anc is not None:
    line = net.lines[(anc,curr)]
    meas = line.p_eq.m
    meas = np.concatenate((meas,line.q_eq.m),axis=0)
    sigma = line.p_eq.sigma**2
    sigma = np.concatenate((sigma,line.q_eq.sigma**2),axis=0)
    W = sigma*np.identity(sigma.shape[0])
    x = np.concatenate((curr.v_states,curr.delta_states),axis=0)
    ref = np.concatenate((anc.v_states,anc.delta_states),axis=0)
    y = line.y_mtx
    err = []
    i = 0
    e = 10
    while np.abs(e) > 0.00001 and i < 100:
      H = jac(x,ref,y).reshape(-1,len(list(x)))
      G = H.T.dot(W).dot(H)
      b = H.T.dot(W).dot(meas-back_equations(x,ref,y,flag))
      dx = nu.linalg.solve(G,b)
      e = np.sum(dx)/len(dx)
      # print(e)
      err.append(e)
      i = i + 1
      x = x + dx
    curr.v_states = x[0:3]
    curr.delta_states = x[3:6]

def forward(net):
  jac = jacobian(fwd_equations)
  fse(net,None,net.buses[1],jac)

def state_estimation(net):
  
  # Putting all states in a vector to monitor the error
  x = init_states([net.buses[1]],np.empty((0,1)))

  i = 0
  err = []
  e = 10
  while e > 0.00001 and i < 100:
    print('Backward',i)
    backward(net)
    print('Forward',i)
    forward(net)
    new_x = init_states([net.buses[1]],np.empty((0,1)))
    dx = np.abs(new_x - x)
    e = sum(dx)/len(dx)
    err = err + [e]
    x = new_x
    i = i + 1
  print_states([net.buses[1]])

if __name__ == '__main__':

  net = Network()

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

  net.add_bus(Bus(1,slack=True,name='Bus 1',v_base=v_base/np.sqrt(3),config='4'),1)
  slack_idx = 1

  # Bus 2 IEEE 4-bus feeder
  net.add_bus(Bus(2,name='Bus 2',v_base=v_base/np.sqrt(3),config='4'),2)
  
  v_base = 4160
  # Bus 3 IEEE 4-bus feeder
  net.add_bus(Bus(3,name='Bus 3',v_base=v_base/np.sqrt(3),config='4'),3)

  # Bus 4 IEEE 4-bus feeder Unbalanced
  # s = A.T.dot(np.array([[1200000/0.9,1200000/0.9,1200000/0.9]]).T*np.exp(1j*np.arccos([[0.9,0.9,0.9]])).T/(s_base/3))
  s = np.array([[1275000/0.85,1800000/0.9,2375000/0.95]]).T*np.exp(1j*np.arccos([[0.85,0.9,0.95]])).T/(s_base/3)
  p = np.real(s)
  q = np.imag(s)
  net.add_bus(Bus(4,p_inj=p,q_inj=q,name='Bus 4',v_base=v_base/np.sqrt(3),config='4'),4)

  # Sort buses by index. Slack always has index 1
  net.buses = {k:v for k,v in sorted(net.buses.items())}

  v_base = 12470
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)

  b = (net.buses[1],net.buses[2])
  l = Line(idx=0,buses=b,z_mtx=zy*f_to_m(2000)/z_base,i_base=i_base,name='Line 1-2',config='4-wire') # 2000ft = 0.3787879 miles
  net.add_line(l,b)

  V1 = 12470/np.sqrt(3)*np.exp(1j*np.array([[0],[-2*np.pi/3],[2*np.pi/3]]))/(v_base/np.sqrt(3))
  # Power Flow from Bus 1 to Bus 2
  V2 = np.array([[7164,7110,7082]]).T*np.exp(1j*np.deg2rad([[-0.1,-120.2,119.3]])).T/(v_base/np.sqrt(3))
  
  P = Pflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),l.y_mtx,'ij')
  Q = Qflow(np.abs(V1),np.angle(V1),np.abs(V2),np.angle(V2),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.02*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.02*np.ones((3,1)))

  v_base = 4160
  z_base = v_base**2/s_base
  i_base = s_base/(np.sqrt(3)*v_base)
  
  b = (net.buses[2],net.buses[3])
  l = Line(idx=1,buses=b,z_mtx=None,i_base=None,trans=True,trans_conf='YgYg',step='down',trans_z=tr,name='Transformer')
  net.add_line(l,b)

  V3 = np.array([[2305,2255,2203]]).T*np.exp(1j*np.deg2rad([[-2.3,-123.6,114.8]])).T/(v_base/np.sqrt(3))
  P = Pflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  Q = Qflow(np.abs(V2),np.angle(V2),np.abs(V3),np.angle(V3),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.02*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.02*np.ones((3,1)))

  b = (net.buses[3],net.buses[4])
  l = Line(idx=2,buses=b,z_mtx=zy*f_to_m(2500)/z_base,i_base=i_base,name='Line 3-4',config='4-wire') # 2500ft = 0.4734848 miles
  net.add_line(l,b)

  V4 = np.array([[2175,1930,1833]]).T*np.exp(1j*np.deg2rad([[-4.1,-126.8,102.8]])).T/(v_base/np.sqrt(3))
  # Power Flow from Bus 3 to Bus 4
  P = Pflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  Q = Qflow(np.abs(V3),np.angle(V3),np.abs(V4),np.angle(V4),l.y_mtx,'ij')
  l.meas['ij'] = {}
  l.meas['ij']['p'] = Measurement(t='PFij',m=P,sigma=0.02*np.ones((3,1)))
  l.meas['ij']['q'] = Measurement(t='QFij',m=Q,sigma=0.02*np.ones((3,1)))
  create_levels(net)
  state_estimation(net)
  # -------------------------------------------------------------------------------------------
    
  # jac = jacobian(equations)
  # err = []
  # i = 0
  # e = 10
  # while np.abs(e) > 0.00001 and i < 100:
  #   print(i)
  #   H = jac(x).reshape(-1,len(list(x)))
  #   # printA(H)
  #   # input()
  #   G = H.T.dot(W).dot(H)
  #   # print(np.linalg.det(H.T.dot(H)))
  #   # input()
  #   b = H.T.dot(W).dot(meas-equations(x))
  #   dx = nu.linalg.solve(G,b)
  #   # print(dx)
  #   e = np.sum(dx)/len(dx)
  #   print(e)
  #   err.append(e)
  #   i = i + 1
  #   if np.abs(e) > 0.00001:
  #     x = x + dx.flatten()
    
  # print(x)
  # for i in range(3):
  #   v_base = 12470 if i == 0 else 4160
  #   print('----- V'+str(i+2)+' -----')
  #   for j,ph in zip(range(3),['a: ','b: ','c: ']):
  #     print(ph,end='')
  #     v = round(x[3+6*i+j]*v_base/np.sqrt(3),3)
  #     d = round(np.degrees(x[6+6*i+j]),3)
  #     print(str(v)+'/'+str(d))